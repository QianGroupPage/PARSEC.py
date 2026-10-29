"""Forces of a spin-polarized periodic cell.

Two Ge atoms (a nonlinear core correction and nonlocal projectors) in a 7.7 bohr cubic cell, Gamma point.  Neutral they
are closed-shell; with one electron removed the cell is magnetic.  The forces of a spin-polarized result are the
formulas of the unpolarized ones on the total density, the channel average of the exchange-correlation potential and the
orbitals of both channels at one electron per state (``forcloc.f90``, ``forcnloc.f90``), so the tests check

* that the periodic system starts the spin loop like the isolated one (``initchrg.f90``) and that the loop reaches the
  energy of the narrower driver ``run_scf_spin_polarized``;
* that a non-magnetic run of the spin loop has the energy and the forces of the unpolarized loop;
* that each force term is the derivative of the spin energy functional at the converged orbitals, which is exact for
  the cubic spline of the tables (no smearing or SCF response enters).
"""

from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.acceleration.V_ion import periodic_setup
from parsec_python.Forces import compute_periodic_forces
from parsec_python.SCF import prepare_periodic_single_point, run_scf, update_periodic_geometry
from parsec_python.SCF.spin_polarized import run_scf_spin_polarized
from parsec_python.V_ion import ewald
from parsec_python.V_ion import ionic_potential as _ionic

_GE = Path(__file__).resolve().parents[1] / "data" / "Ge_POTRE.DAT"
_POSITIONS = np.array([[2.4, 3.1, 3.9], [5.0, 4.4, 3.3]])


def _problem(*, net_charge: float, spin: bool, polarization: float = 0.1) -> SinglePointInput:
    return SinglePointInput(
        atoms=[Atom("Ge", position.copy()) for position in _POSITIONS],
        pseudopotentials={"Ge": SpeciesPotential(_GE, 1, use_spline=True, initial_spin_polarization=polarization)},
        grid=PeriodicGridSettings(spacing=0.7, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=7.7 * np.eye(3)),
        scf=SCFSettings(
            max_iterations=80,
            number_of_states=6,
            net_charge=net_charge,
            spin_polarized=spin,
            convergence_criterion=1.0e-9,
            fermi_temperature_kelvin=300.0,
        ),
        hartree=HartreeSettings(relative_tolerance=1e-10, absolute_tolerance=1e-13, max_iterations=4000),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-9),
    )


class PeriodicSpinForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # the fast periodic setup gives the same potentials as the dense one (to 1e-7 Ry) in a second instead of minutes
        cls._saved = (ewald._local_potential_provider, _ionic._support_finder)
        periodic_setup.install()
        cls.magnetic_system = prepare_periodic_single_point(_problem(net_charge=1.0, spin=True))
        cls.magnetic = run_scf(cls.magnetic_system)
        cls.spin_system = prepare_periodic_single_point(_problem(net_charge=0.0, spin=True, polarization=0.0))
        cls.spin_closed = run_scf(cls.spin_system)
        cls.plain_system = prepare_periodic_single_point(_problem(net_charge=0.0, spin=False))
        cls.plain = run_scf(cls.plain_system)

    @classmethod
    def tearDownClass(cls) -> None:
        ewald._local_potential_provider, _ionic._support_finder = cls._saved

    def test_the_system_starts_like_initchrg(self) -> None:
        up, down = self.magnetic_system.initial_spin_densities
        # the two channels add up to twice the (normalized) total density; the difference is the polarization
        np.testing.assert_allclose(up + down, 2.0 * self.magnetic_system.initial_density, rtol=1e-12)
        self.assertGreater(float(np.sum(up - down)), 0.0)
        self.assertIsNone(self.plain_system.initial_spin_densities)

    def test_the_spin_loop_is_magnetic_and_reaches_the_energy_of_the_narrow_driver(self) -> None:
        self.assertTrue(self.magnetic.converged)
        up, down = self.magnetic.spin
        moment = (np.sum(up.density) - np.sum(down.density)) * self.magnetic_system.grid.volume_element
        self.assertGreater(moment, 0.5)
        narrow = run_scf_spin_polarized(self.magnetic_system)
        self.assertAlmostEqual(self.magnetic.energies.total, narrow.energies.total, delta=1.0e-6)
        self.assertNotEqual(self.magnetic.energies.ion_ion, 0.0)

    def test_a_nonmagnetic_spin_run_has_the_energy_and_forces_of_the_unpolarized_run(self) -> None:
        self.assertTrue(self.spin_closed.converged and self.plain.converged)
        self.assertAlmostEqual(self.spin_closed.energies.total, self.plain.energies.total, delta=1.0e-6)
        spin = compute_periodic_forces(self.spin_system, self.spin_closed)
        plain = compute_periodic_forces(self.plain_system, self.plain)
        self.assertGreater(float(np.max(np.abs(plain.total))), 0.1)
        np.testing.assert_allclose(spin.total, plain.total, atol=2.0e-6)
        np.testing.assert_allclose(spin.nonlocal_pseudopotential, plain.nonlocal_pseudopotential, atol=2.0e-6)
        np.testing.assert_allclose(spin.local.core_correction, plain.local.core_correction, atol=2.0e-6)

    def test_each_force_term_is_the_derivative_of_the_spin_functional(self) -> None:
        system, result = self.magnetic_system, self.magnetic
        up, down = result.spin
        dv = system.grid.volume_element
        forces = compute_periodic_forces(system, result)
        reference = {
            "ion_ion": forces.ion_ion,
            "ionic": forces.local.ionic,
            "core": forces.local.core_correction,
            "nonlocal": forces.nonlocal_pseudopotential,
        }
        density = up.density + down.density

        def terms(atoms) -> dict[str, float]:
            moved = update_periodic_geometry(system, atoms)
            free = moved.hamiltonian(np.zeros(moved.grid.size)).as_linear_operator()
            nonlocal_energy = 0.0
            for channel in result.spin:  # one electron per state of a channel
                psi = channel.wavefunctions
                nonlocal_energy += float(
                    np.asarray(channel.occupations)
                    @ np.einsum("ij,ij->j", psi, free @ psi - moved.negative_laplacian @ psi)
                )
            return {
                "ion_ion": moved.ion_ion_energy,
                "ionic": dv * float(density @ moved.ionic_potential),
                "core": moved.evaluate_spin_xc(up.density, down.density).total_energy,
                "nonlocal": nonlocal_energy,
            }

        def shifted(atom: int, axis: int, step: float):
            atoms = [Atom(a.symbol, np.array(a.position)) for a in system.atoms]
            atoms[atom].position[axis] += step
            return atoms

        tolerance = {"ion_ion": 1.0e-8, "ionic": 1.0e-6, "core": 2.0e-3, "nonlocal": 1.0e-6}
        for atom, axis in ((0, 0), (1, 2), (0, 1)):
            def central(step: float) -> dict[str, float]:
                plus, minus = terms(shifted(atom, axis, +step)), terms(shifted(atom, axis, -step))
                return {key: -(plus[key] - minus[key]) / (2.0 * step) for key in plus}

            fine, coarse = central(0.0025), central(0.005)
            for key, expected in reference.items():
                numeric = (4.0 * fine[key] - coarse[key]) / 3.0
                with self.subTest(term=key, atom=atom, axis=axis):
                    self.assertAlmostEqual(float(expected[atom, axis]), numeric, delta=tolerance[key])
        # the terms are not all negligible: the check bites
        for key in ("ionic", "nonlocal", "core"):
            self.assertGreater(float(np.max(np.abs(reference[key]))), 1.0e-2)

    def test_the_net_force_is_removed_and_there_is_no_torque(self) -> None:
        forces = compute_periodic_forces(self.magnetic_system, self.magnetic)
        np.testing.assert_allclose(forces.total.sum(axis=0), 0.0, atol=1.0e-12)
        np.testing.assert_array_equal(forces.torque, 0.0)


if __name__ == "__main__":
    unittest.main()
