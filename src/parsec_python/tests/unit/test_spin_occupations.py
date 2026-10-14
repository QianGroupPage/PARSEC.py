"""Occupations, residuals and energy bookkeeping of two spin channels.

``flevel.f90`` fills both channels up to one Fermi level, ``getsre.f90``
measures the residual of each channel and ``totnrg.f90`` sums the band
energy and the double-counting integrals over them.  The Fermi levels PARSEC
printed for the levels of four of its spin-polarized runs are reproduced
here; everything else is compared with the unpolarized routines, which two
identical channels must reduce to.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from parsec_python.Energy import spin_total_energy, total_energy
from parsec_python.Mixer import (
    potential_residual_metrics,
    spin_potential_residual_metrics,
)
from parsec_python.Occupations import (
    density_from_orbitals,
    fermi_occupations,
    read_spin_occupation_file,
    spin_density_from_orbitals,
    spin_fermi_occupations,
    spin_fixed_occupations,
)
from parsec_python.tests.martins_old_reference import DATA


# Levels (Ry) of the last SCF step of Fortran PARSEC runs with a Fermi
# temperature of 80 K: up channel, down channel, electrons, the Fermi level
# and the electrons per channel it printed.
FORTRAN_LEVELS = {
    "N atom": (
        (-1.4379732858, -0.6001602286, -0.6001602286, -0.6001602286,
         0.3267568658, 0.5477391593, 0.5477391593, 0.5477391593),
        (-1.0935407700, -0.2684781120, -0.2684781120, -0.2684781120,
         0.4159760111, 0.6668519501, 0.6668519501, 0.6668519501),
        5.0,
        -0.3856,
        (4.0, 1.0),
    ),
    "CH3": (
        (-1.2024740773, -0.7469316089, -0.7004062720, -0.3623054688,
         0.2960163441, 0.3727524588, 0.4212554323, 0.5049514092),
        (-1.1385055970, -0.7149910612, -0.6693394683, -0.2118284016,
         0.3289970117, 0.3901662015, 0.4359489501, 0.5546781136),
        7.0,
        -0.3239,
        (4.0, 3.0),
    ),
}


class SpinFermiLevelTests(unittest.TestCase):
    def test_fermi_levels_of_fortran_runs(self) -> None:
        for name, (up, down, electrons, level, per_channel) in FORTRAN_LEVELS.items():
            with self.subTest(system=name):
                result = spin_fermi_occupations(up, down, electrons, 80.0)
                # Four decimals are printed.
                self.assertAlmostEqual(result.fermi_level, level, delta=5.01e-5)
                self.assertEqual(result.electrons, per_channel)
                self.assertEqual(
                    result.net_magnetic_moment, per_channel[0] - per_channel[1]
                )
                self.assertAlmostEqual(result.electron_count, electrons, delta=1.0e-12)

    def test_identical_channels_fill_like_one_unpolarized_channel(self) -> None:
        levels = np.array([-1.2, -0.61, -0.6, -0.59, -0.2, 0.1])
        for temperature in (80.0, 3000.0):
            for electrons in (4.0, 7.0, 7.3):
                with self.subTest(temperature=temperature, electrons=electrons):
                    spin = spin_fermi_occupations(levels, levels, electrons, temperature)
                    plain = fermi_occupations(levels, electrons, temperature)
                    np.testing.assert_allclose(
                        spin.occupations[0], plain.occupations, rtol=0, atol=2.0e-11
                    )
                    np.testing.assert_array_equal(
                        spin.occupations[0], spin.occupations[1]
                    )
                    self.assertAlmostEqual(spin.electron_count, electrons, delta=1.0e-12)
                    self.assertEqual(spin.net_magnetic_moment, 0.0)

    def test_one_level_for_both_channels_at_high_temperature(self) -> None:
        up = np.array([-0.60, -0.30, 0.10])
        down = np.array([-0.55, -0.20, 0.15])
        result = spin_fermi_occupations(up, down, 3.0, 20000.0)
        self.assertAlmostEqual(result.electron_count, 3.0, delta=1.0e-12)
        kbt = 6.33327186e-6 * 20000.0
        for levels, occupations in zip((up, down), result.occupations):
            np.testing.assert_allclose(
                occupations,
                1.0 / (np.exp((levels - result.fermi_level) / kbt) + 1.0),
                rtol=1.0e-12,
            )
        self.assertGreater(result.net_magnetic_moment, 0.0)

    def test_zero_temperature_fills_the_merged_levels(self) -> None:
        up = np.array([-1.0, -0.5, -0.5, 0.2])
        down = np.array([-0.9, -0.5, 0.0, 0.3])
        result = spin_fermi_occupations(up, down, 4.0, 0.0)
        # -1.0, -0.9, then two electrons for the three levels at -0.5.
        np.testing.assert_allclose(result.occupations[0], [1.0, 2.0 / 3.0, 2.0 / 3.0, 0.0])
        np.testing.assert_allclose(result.occupations[1], [1.0, 2.0 / 3.0, 0.0, 0.0])
        self.assertAlmostEqual(result.electron_count, 4.0, places=12)
        full = spin_fermi_occupations(up, down, 2.0, 0.0)
        np.testing.assert_array_equal(full.occupations[0], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(full.occupations[1], [1.0, 0.0, 0.0, 0.0])

    def test_many_levels_within_kt_of_the_fermi_level(self) -> None:
        """Double precision cannot place such a level to 1e-12 of an electron.

        The bisection then stops moving; ``flevel.f90`` runs out of steps and
        reports "Fermi level not found".
        """

        for seed in range(10):
            generator = np.random.default_rng(seed)
            channels = [
                np.concatenate(
                    (
                        np.sort(generator.uniform(-1.0, -0.5, 2000)),
                        np.sort(generator.uniform(-0.302, -0.298, 500)),
                    )
                )
                for _ in range(2)
            ]
            with self.subTest(seed=seed):
                result = spin_fermi_occupations(channels[0], channels[1], 4500.0, 80.0)
                self.assertAlmostEqual(result.electron_count, 4500.0, delta=1.0e-8)
                self.assertLess(abs(result.fermi_level + 0.3), 2.0e-3)

    def test_refusals(self) -> None:
        levels = np.array([-1.0, -0.5])
        with self.assertRaisesRegex(ValueError, "do not fit the 4 states"):
            spin_fermi_occupations(levels, levels, 4.5)
        # flevel.f90 error 422: no empty state above the electrons.
        with self.assertRaisesRegex(RuntimeError, "increase States_Num"):
            spin_fermi_occupations(levels, levels, 4.0)
        with self.assertRaisesRegex(ValueError, "must be sorted"):
            spin_fermi_occupations(levels[::-1], levels, 1.0)
        with self.assertRaisesRegex(ValueError, "selects fixed occupations"):
            spin_fermi_occupations(levels, levels, 1.0, -1.0)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            spin_fermi_occupations(levels, np.empty(0), 1.0)


class SpinFixedOccupationTests(unittest.TestCase):
    def test_each_channel_takes_its_section(self) -> None:
        up = np.array([-0.5, 0.1, 0.2])
        down = np.array([-0.4, 0.2, 0.3])
        result = spin_fixed_occupations(up, down, 1.0, (0.7, 0.0, 0.0), (0.3, 0.0, 0.0))
        np.testing.assert_array_equal(result.occupations[0], [0.7, 0.0, 0.0])
        np.testing.assert_array_equal(result.occupations[1], [0.3, 0.0, 0.0])
        self.assertTrue(np.isnan(result.fermi_level))
        self.assertAlmostEqual(result.net_magnetic_moment, 0.4, places=15)
        # initial.f90 error 233: the sum over both channels is the count.
        with self.assertRaisesRegex(ValueError, "hold 1.400000 electrons"):
            spin_fixed_occupations(up, down, 1.0, (0.7, 0.0, 0.0), (0.7, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "2 fixed occupations were given"):
            spin_fixed_occupations(up, down, 1.0, (0.7, 0.3), (0.0, 0.0, 0.0))

    def test_two_sections_of_occup_in(self) -> None:
        # The file of the Fortran example examples/tests/relaxation.
        example = read_spin_occupation_file(DATA / "sina4_occup.in", 8)
        self.assertEqual(example.occupations_up, (1.0,) * 4 + (0.0,) * 4)
        self.assertEqual(example.occupations_down, (1.0,) * 3 + (0.5, 0.5) + (0.0,) * 3)
        self.assertEqual(example.unread_lines, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "occup.in"
            path.write_text("up\n0\n1\n1 0.7\ndown\n0\n1\n1 0.3d0\nrest\n")
            result = read_spin_occupation_file(path, 4)
            self.assertEqual(result.occupations_up, (0.7, 0.0, 0.0, 0.0))
            self.assertEqual(result.occupations_down, (0.3, 0.0, 0.0, 0.0))
            self.assertEqual(result.unread_lines, 1)
            path.write_text("up\n1\n0\n")
            with self.assertRaisesRegex(ValueError, "section of spin channel 2"):
                read_spin_occupation_file(path, 4)
            path.write_text("up\n1\n0\ndown\n")
            with self.assertRaisesRegex(ValueError, "ends before the number"):
                read_spin_occupation_file(path, 4)


class SpinDensityResidualEnergyTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(5)
        self.volume_element = 0.064
        self.orbitals, _ = np.linalg.qr(generator.normal(size=(60, 5)))
        self.occupations = np.array([1.0, 1.0, 0.6, 0.25, 0.0])
        self.generator = generator

    def test_channel_density_holds_one_electron_per_state(self) -> None:
        density = spin_density_from_orbitals(
            self.orbitals, self.occupations, self.volume_element
        )
        self.assertAlmostEqual(
            self.volume_element * float(np.sum(density)),
            float(np.sum(self.occupations)),
            places=12,
        )
        # A backend builder that forms the density of doubly occupied
        # orbitals gives the same array for half the occupations.
        np.testing.assert_array_equal(
            density,
            density_from_orbitals(
                self.orbitals, 0.5 * self.occupations, self.volume_element
            ),
        )

    def test_residuals_of_each_channel(self) -> None:
        size = self.orbitals.shape[0]
        inputs = (self.generator.normal(size=size), self.generator.normal(size=size))
        outputs = (
            inputs[0] + 0.01 * self.generator.normal(size=size),
            inputs[1] + 0.02 * self.generator.normal(size=size),
        )
        densities = (
            spin_density_from_orbitals(self.orbitals, self.occupations, self.volume_element),
            spin_density_from_orbitals(
                self.orbitals, self.occupations[::-1].copy(), self.volume_element
            ),
        )
        electrons = (float(np.sum(self.occupations)),) * 2
        metrics = spin_potential_residual_metrics(
            inputs, outputs, densities, self.volume_element, electrons
        )
        for channel in range(2):
            plain = potential_residual_metrics(
                inputs[channel],
                outputs[channel],
                densities[channel],
                self.volume_element,
                electrons[channel],
            )
            self.assertAlmostEqual(
                metrics.weighted_by_spin[channel], plain.weighted, places=15
            )
            self.assertAlmostEqual(metrics.plain_by_spin[channel], plain.plain, places=15)
        # parsec.F90 tests the larger of the two.
        self.assertEqual(metrics.weighted, max(metrics.weighted_by_spin))
        self.assertEqual(metrics.plain, max(metrics.plain_by_spin))
        # getsre.f90: no weighted residual for a channel without electrons.
        empty = spin_potential_residual_metrics(
            inputs,
            outputs,
            (densities[0], np.zeros(size)),
            self.volume_element,
            (electrons[0], 0.0),
        )
        self.assertEqual(empty.weighted_by_spin[1], 0.0)
        self.assertGreater(empty.plain_by_spin[1], 0.0)

    def test_identical_channels_give_the_unpolarized_energy(self) -> None:
        size = self.orbitals.shape[0]
        eigenvalues = np.array([-1.1, -0.7, -0.6, -0.2, 0.3])
        channel_density = spin_density_from_orbitals(
            self.orbitals, self.occupations, self.volume_element
        )
        ionic = -np.abs(self.generator.normal(size=size))
        input_potential = ionic + self.generator.normal(size=size)
        hartree = np.abs(self.generator.normal(size=size))
        xc = -np.abs(self.generator.normal(size=size))
        spin = spin_total_energy(
            (eigenvalues, eigenvalues),
            (self.occupations, self.occupations),
            (channel_density, channel_density),
            (input_potential, input_potential),
            ionic,
            hartree,
            (xc, xc),
            -3.21,
            1.5,
            self.volume_element,
        )
        plain = total_energy(
            eigenvalues,
            self.occupations,
            2.0 * channel_density,
            input_potential,
            ionic,
            hartree,
            xc,
            -3.21,
            1.5,
            self.volume_element,
        )
        for name in (
            "eigenvalue",
            "hartree",
            "integral_vxc_rho",
            "exchange_correlation",
            "electron_ion",
            "ion_ion",
            "electronic",
            "total",
        ):
            self.assertAlmostEqual(
                getattr(spin, name), getattr(plain, name), places=12, msg=name
            )

    def test_energy_sums_over_different_channels(self) -> None:
        size = self.orbitals.shape[0]
        densities = (
            spin_density_from_orbitals(self.orbitals, self.occupations, self.volume_element),
            spin_density_from_orbitals(
                self.orbitals, np.array([1.0, 0.5, 0.0, 0.0, 0.0]), self.volume_element
            ),
        )
        ionic = self.generator.normal(size=size)
        inputs = (self.generator.normal(size=size), self.generator.normal(size=size))
        hartree = self.generator.normal(size=size)
        xc = (self.generator.normal(size=size), self.generator.normal(size=size))
        eigenvalues = (np.array([-1.0, -0.5, -0.4, -0.1, 0.2]), np.array([-0.9, -0.3, 0.0, 0.1, 0.4]))
        occupations = (self.occupations, np.array([1.0, 0.5, 0.0, 0.0, 0.0]))
        energies = spin_total_energy(
            eigenvalues, occupations, densities, inputs, ionic, hartree, xc,
            -2.0, 0.75, self.volume_element,
        )
        h3 = self.volume_element
        band = float(occupations[0] @ eigenvalues[0] + occupations[1] @ eigenvalues[1])
        total = densities[0] + densities[1]
        old = h3 * float(
            densities[0] @ (inputs[0] - ionic) + densities[1] @ (inputs[1] - ionic)
        )
        self.assertAlmostEqual(energies.eigenvalue, band, places=13)
        self.assertAlmostEqual(energies.hartree, 0.5 * h3 * float(total @ hartree), places=13)
        self.assertAlmostEqual(
            energies.integral_vxc_rho,
            h3 * float(densities[0] @ xc[0] + densities[1] @ xc[1]),
            places=13,
        )
        self.assertAlmostEqual(energies.electron_ion, h3 * float(total @ ionic), places=13)
        self.assertAlmostEqual(
            energies.total, band - old + energies.hartree - 2.0 + 0.75, places=12
        )


if __name__ == "__main__":
    unittest.main()
