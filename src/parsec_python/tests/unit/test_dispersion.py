"""Component tests of the Tkatchenko-Scheffler correction (``efvdw.f90``).

The pair sum is compared with a loop written after the Fortran and its forces
with finite differences of its energy; the Hirshfeld volumes with a direct
evaluation of ``initchrg_f.f90`` on a small grid.  The comparison of whole
runs with Fortran PARSEC is in the report of the parity round.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from parsec_python.Energy.dispersion import (
    FREE_ATOM_DATA,
    PERIODIC_SHELL_TOLERANCE,
    HirshfeldVolumes,
    damping_function,
    hirshfeld_volumes,
    ts_dispersion,
    ts_pair_energy,
)
from parsec_python.Energy.total_energy import with_dispersion
from parsec_python.Grid import build_cluster_grid
from parsec_python.Output.correction_output import dispersion_report_lines, list_directed_real
from parsec_python.V_ion import load_pseudopotentials, superpose_atomic_density
from parsec_python.models import Atom, EnergyBreakdown, GridSettings, SpeciesPotential


DATA = Path(__file__).resolve().parents[1] / "data"


def _efvdw_loop(positions, c6, alpha, r0):
    """The cluster part of ``efvdw.f90`` with its loops (Ry, Ha/bohr)."""

    count = len(positions)
    energy = 0.0
    forces = np.zeros((count, 3))
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            c6ij = 2 * c6[i] * c6[j] / (alpha[j] / alpha[i] * c6[i] + alpha[i] / alpha[j] * c6[j])
            r = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
            rij = r0[i] + r0[j]
            f = 1.0 / (1.0 + np.exp(-20.0 * (r / (0.94 * rij) - 1.0)))
            add = f * c6ij / r**6
            energy += add
            rr0 = rij * 0.94
            rexp = np.exp(-20.0 * (r / (0.94 * rij) - 1.0))
            rfac = -6.0 / r + 20.0 * rexp / ((1.0 + rexp) * rr0)
            ffac = -add * rfac / r
            forces[i] -= (positions[i] - positions[j]) * ffac
    return 2.0 * (-0.5 * energy), forces


class DampingTest(unittest.TestCase):
    def test_limits(self):
        self.assertAlmostEqual(float(damping_function(0.94 * 7.0, 7.0)), 0.5, places=15)
        self.assertLess(float(damping_function(1.0, 7.0)), 1.0e-6)
        self.assertGreater(float(damping_function(30.0, 7.0)), 1.0 - 1.0e-15)


class PairSumTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.positions = rng.uniform(-4.0, 4.0, size=(5, 3))
        self.c6 = rng.uniform(5.0, 50.0, size=5)
        self.alpha = rng.uniform(3.0, 15.0, size=5)
        self.radius = rng.uniform(2.8, 3.8, size=5)

    def test_cluster_sum_follows_efvdw(self):
        energy, forces, shells = ts_pair_energy(self.positions, self.c6, self.alpha, self.radius)
        reference, reference_forces = _efvdw_loop(self.positions, self.c6, self.alpha, self.radius)
        self.assertAlmostEqual(energy, reference, places=14)
        np.testing.assert_allclose(forces, 2.0 * reference_forces, rtol=1e-12, atol=1e-16)
        self.assertEqual(shells, 0)
        # Identical atoms combine to their own C6.
        same, _, _ = ts_pair_energy(
            np.array([[0, 0, 0], [0, 0, 8.0]]), np.full(2, 20.0), np.full(2, 9.0), np.full(2, 3.2)
        )
        self.assertAlmostEqual(
            same, -20.0 * float(damping_function(8.0, 6.4)) / 8.0**6 * 2.0, places=15
        )

    def test_forces_are_the_derivative_of_the_energy(self):
        for lattice in (None, np.diag([9.0, 10.0, 11.0])):
            _, forces, _ = ts_pair_energy(self.positions, self.c6, self.alpha, self.radius, lattice)
            step = 1.0e-5
            for atom in range(2):
                for component in range(3):
                    plus = self.positions.copy()
                    minus = self.positions.copy()
                    plus[atom, component] += step
                    minus[atom, component] -= step
                    numeric = -(
                        ts_pair_energy(plus, self.c6, self.alpha, self.radius, lattice)[0]
                        - ts_pair_energy(minus, self.c6, self.alpha, self.radius, lattice)[0]
                    ) / (2 * step)
                    self.assertAlmostEqual(forces[atom, component], numeric, delta=1e-7 + 1e-5 * abs(numeric))

    def test_periodic_shells_stop_below_the_threshold(self):
        lattice = np.diag([8.0, 8.0, 8.0])
        energy, forces, shells = ts_pair_energy(
            self.positions % 8.0, self.c6, self.alpha, self.radius, lattice
        )
        self.assertGreater(shells, 1)
        # The last shell is below the threshold and the one before it is not.
        cluster, _, _ = ts_pair_energy(self.positions % 8.0, self.c6, self.alpha, self.radius)
        self.assertLess(energy, cluster)
        self.assertLess(abs(float(np.sum(forces))), 1.0e-12)
        self.assertEqual(PERIODIC_SHELL_TOLERANCE, 7.3e-8)


class FreeAtomTableTest(unittest.TestCase):
    def test_table_is_that_of_efvdw(self):
        candidates = []
        if os.environ.get("PARSEC_FORTRAN_SOURCE"):
            candidates.append(Path(os.environ["PARSEC_FORTRAN_SOURCE"]) / "efvdw.f90")
        for parent in Path(__file__).resolve().parents:
            candidates.append(parent / ".tmp" / "fortran_parity_20261027" / "src" / "efvdw.f90")
            candidates.append(parent / "src" / "efvdw.f90")
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            self.skipTest("PARSEC source efvdw.f90 not available")
        pattern = re.compile(
            r"clust%name\(itype\)=='(\w+)'\)\s*then\s*alpha_free\(ja\)=([\d.]+)D0\s*"
            r"c6ii_free\(ja\)=([\d.]+)D0\s*R0\(ja\)=([\d.]+)D0",
            re.S,
        )
        rows = {name: (float(a), float(c), float(r)) for name, a, c, r in pattern.findall(source.read_text())}
        self.assertEqual(rows, FREE_ATOM_DATA)


class HirshfeldTest(unittest.TestCase):
    def setUp(self):
        self.specifications = {
            "O": SpeciesPotential(DATA / "water_O_POTRE.DAT", 1, element_symbol="O"),
            "H": SpeciesPotential(DATA / "water_H_POTRE.DAT", 0, element_symbol="H"),
        }
        self.potentials = load_pseudopotentials(self.specifications)
        self.grid = build_cluster_grid(GridSettings(radius=6.0, spacing=0.5))
        self.atoms = (
            Atom("O", [0.0, 0.0, 0.1]),
            Atom("H", [0.0, 1.4, -0.9]),
            Atom("H", [0.0, -1.4, -0.9]),
        )

    def test_volumes_follow_initchrg_f(self):
        grid = self.grid
        density = superpose_atomic_density(grid, self.atoms, self.potentials, self.specifications)
        density *= 1.1
        volumes = hirshfeld_volumes(grid, self.atoms, self.potentials, self.specifications, density)
        free_densities = []
        distances = []
        for atom in self.atoms:
            values = superpose_atomic_density(grid, (atom,), self.potentials, self.specifications)
            values *= self.potentials[atom.symbol].ionic_charge / (np.sum(values) * grid.volume_element)
            free_densities.append(values)
            distances.append(np.linalg.norm(grid.coordinates - atom.position, axis=1))
        total = np.sum(free_densities, axis=0)
        for index in range(3):
            free = np.sum(distances[index] ** 3 * free_densities[index]) * grid.volume_element
            effective = (
                np.sum(distances[index] ** 3 * free_densities[index] / total * density)
                * grid.volume_element
            )
            self.assertAlmostEqual(volumes.free[index], free, places=10)
            self.assertAlmostEqual(volumes.effective[index], effective, places=10)

    def test_free_atom_has_ratio_one(self):
        atom = (Atom("O", [0.0, 0.0, 0.0]),)
        density = superpose_atomic_density(self.grid, atom, self.potentials, self.specifications)
        density *= 6.0 / (np.sum(density) * self.grid.volume_element)
        volumes = hirshfeld_volumes(self.grid, atom, self.potentials, self.specifications, density)
        self.assertAlmostEqual(float(volumes.ratio[0]), 1.0, places=12)
        result = ts_dispersion(atom, ["O"], volumes)
        self.assertAlmostEqual(float(result.c6[0]), FREE_ATOM_DATA["O"][1], places=10)
        self.assertEqual(result.energy, 0.0)

    def test_unknown_element_is_refused(self):
        volumes = HirshfeldVolumes(free=np.ones(1), effective=np.ones(1))
        with self.assertRaises(ValueError):
            ts_dispersion((Atom("X", [0, 0, 0]),), ["Xx"], volumes)


class DispersionReportTest(unittest.TestCase):
    def test_list_directed_reals_are_those_of_gfortran(self):
        # Printed by gfortran 13 with write(*,*).
        samples = {
            0.0123: "   1.2300000000000000E-002 ",
            12.3: "   12.300000000000001      ",
            -1.5: "  -1.5000000000000000      ",
            0.0: "   0.0000000000000000      ",
            1.234567890123456789e-5: "   1.2345678901234568E-005 ",
            -0.1: " -0.10000000000000001      ",
            0.09999: "   9.9989999999999996E-002 ",
            123456789.123: "   123456789.12300000      ",
            1.0e17: "   1.0000000000000000E+017 ",
        }
        for value, text in samples.items():
            # One blank more: the separator of the character item after it.
            self.assertEqual(list_directed_real(value) + " ", text)
        row = "".join(list_directed_real(value) for value in (1.0, 2.0, -3.5e-3))
        self.assertEqual(
            row.rstrip(),
            "   1.0000000000000000        2.0000000000000000       -3.5000000000000001E-003",
        )

    def test_energies_and_report(self):
        energies = EnergyBreakdown(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -10.0, -4.0)
        corrected = with_dispersion(energies, -0.002)
        self.assertAlmostEqual(corrected.total, -4.002, places=15)
        self.assertEqual(corrected.electronic, -10.0)
        volumes = HirshfeldVolumes(free=np.array([10.0, 2.0]), effective=np.array([9.0, 1.5]))
        dispersion = ts_dispersion(
            (Atom("O", [0, 0, 0]), Atom("H", [0, 0, 4.0])), ["O", "H"], volumes
        )
        result = type("R", (), {"dispersion": dispersion, "energies": corrected})()
        lines = dispersion_report_lines(result, 2)
        self.assertIn("     TOTAL ENERGY REPORT WITH TS-VDW CORRECTIONS", lines)
        self.assertIn(f"   TS-vDW correction (included)  = {-0.002:20.8f} [Ry]", lines)


class DispersionInputTest(unittest.TestCase):
    INPUT = """
Boundary_Conditions: cluster
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.8
Coordinate_Unit: Cartesian_Bohr
States_Num: 6
Correlation_Type: {correlation}
Ignore_Symmetry: {ignore}
Van_Der_Waals: .true.
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.0 0.0 -0.7
  0.0 0.0 0.7
end Atom_Coord
"""

    def parse(self, correlation="pbe", ignore=".true.", extra=""):
        from parsec_python.Input import parse_parsec_input

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parsec.in"
            path.write_text(self.INPUT.format(correlation=correlation, ignore=ignore, extra=extra))
            (Path(directory) / "H_POTRE.DAT").write_bytes((DATA / "spin_pbe_H_POTRE.DAT").read_bytes())
            return parse_parsec_input(path)

    def test_validation_of_usrinputfile(self):
        from parsec_python.Input import ParsecInputError

        translation = self.parse()
        self.assertTrue(translation.problem.scf.van_der_waals)
        with self.assertRaises(ParsecInputError):
            self.parse(correlation="ca")
        with self.assertRaises(ParsecInputError):
            self.parse(ignore=".false.")
        warned = self.parse(extra="Hf_One: 1")
        self.assertTrue(any("Hf_One" in warning for warning in warned.warnings))
        self.assertEqual(warned.problem.scf.hirshfeld_cells, (0, 0, 0))

    def test_accelerated_cli_keeps_clusters(self):
        from parsec_python.acceleration.cli import _reference_only_features

        # A cluster runs on the accelerated driver (acceleration.Energy).
        self.assertEqual(
            _reference_only_features(self.parse(), accelerated_backend_selected=False), []
        )


if __name__ == "__main__":
    unittest.main()
