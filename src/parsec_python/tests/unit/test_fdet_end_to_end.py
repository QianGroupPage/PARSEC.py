"""FDET of PARSEC 1.5 end to end on the reference path, against the 1.5 binary.

The sample file ``tests/data/fdet_H2_sample_POT.DAT`` is the ``pot.dat`` that
PARSEC 1.5 (parsec-ubuntu_gnu-mpifort-gfortran-.mpi, 2 ranks) wrote for H2
at (+-0.7, 0, 0): sphere 5
bohr, h 0.5 bohr, CA-LDA, ``Ignore_Symmetry``; ``fdet_H_POTRE.DAT`` is the H
pseudopotential of 1.5's test suite.  The tip is H2 at (0.1, 0, 1.9) and
(0.1, 0, 3.3) on the same grid with the sample ions as point charges
(``tip_tf``: Thomas-Fermi, the default).  Fortran's numbers are those of its
``parsec.out``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Forces import compute_forces
from parsec_python.Input import parse_parsec_input
from parsec_python.models import EmbeddingEnergyBreakdown
from parsec_python.Output.pot_dat import potential_field_from_result, read_potential_field
from parsec_python.SCF import prepare_single_point, run_scf

DATA = Path(__file__).resolve().parents[1] / "data"

TIP = """\
Boundary_Sphere_Radius   5.0
Grid_Spacing             0.5
Ignore_Symmetry          true
States_Num               4
Correlation_Type         ca
Convergence_Criterion    1d-7 ry
Max_Iter                 60
Hartree_Boundary_Tolerance off
Potential_Field            true
Potential_Field_Name       sample.dat
Kinetic_Energy_Functional  tf
Add_Point_Charges  true
Point_Typ_Num      1
Pt_Chg             1.0 e
begin Point_Coord
 -0.70 0.0 0.0
  0.70 0.0 0.0
end Point_Coord
Atom_Types_Num           1
Atom_Type        H
Local_Component  s
begin Atom_Coord
 0.10 0.0 1.90
 0.10 0.0 3.30
end Atom_Coord
"""

# parsec.out of PARSEC 1.5 (tip_tf), Ry and Ry/bohr.
FORTRAN = {
    "total": -6.26328247,
    "eigenvalue": -1.23053082,
    "hartree": 2.88944858,
    "integral_vxc_rho": -1.96815424,
    "exchange_correlation": -2.98776932,
    "electron_ion": -8.19777781,
    "ion_ion": 4.58777001,
    "nonadditive_kinetic": 0.27637703,
    "integral_vnad_rho": 0.23221616,
    "tip_ion_sample_electron": -3.16770315,
    "sample_hartree": 2.84032184,
    "sample_electron_ion": -8.22116066,
    "sample_kinetic": 2.79292302,
}
FORTRAN_NET_FORCE = (-1.838918, 0.0, 0.248902)
# "coordinates and tot. forces (after setting net force to zero)"; FDET keeps the net force.
FORTRAN_FORCES_WITHOUT_NET = ((0.347214, 0.0, 0.480444), (-0.347214, 0.0, -0.480444))


class FDETEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._folder = tempfile.TemporaryDirectory()
        directory = Path(cls._folder.name)
        (directory / "parsec.in").write_text(TIP, encoding="utf-8")
        (directory / "H_POTRE.DAT").write_bytes((DATA / "fdet_H_POTRE.DAT").read_bytes())
        (directory / "sample.dat").write_bytes((DATA / "fdet_H2_sample_POT.DAT").read_bytes())
        translation = parse_parsec_input(directory / "parsec.in")
        cls.system = prepare_single_point(translation.problem)
        cls.result = run_scf(cls.system)
        cls.forces = compute_forces(cls.system, cls.result, symmetrize=False)

    @classmethod
    def tearDownClass(cls):
        cls._folder.cleanup()

    def test_energies_equal_parsec_1_5(self):
        energies = self.result.energies
        self.assertTrue(self.result.converged)
        self.assertIsInstance(energies, EmbeddingEnergyBreakdown)
        for name, value in FORTRAN.items():
            with self.subTest(term=name):
                self.assertAlmostEqual(getattr(energies, name), value, delta=1.0e-6)
        self.assertAlmostEqual(energies.total, energies.electronic + energies.ion_ion, places=12)

    def test_forces_keep_the_net_force(self):
        self.assertTrue(self.forces.net_force_kept)
        np.testing.assert_allclose(self.forces.net_force, FORTRAN_NET_FORCE, atol=2.0e-6)
        np.testing.assert_array_equal(self.forces.total, self.forces.symmetrized)
        mean = self.forces.net_force / 2.0
        np.testing.assert_allclose(self.forces.total - mean, FORTRAN_FORCES_WITHOUT_NET, atol=2.0e-6)

    def test_field_and_point_charges(self):
        system = self.system
        field = system.embedding
        # the frozen field is in the ionic potential, the point charges are not
        sample = read_potential_field(DATA / "fdet_H2_sample_POT.DAT")
        self.assertEqual(field.kinetic_energy, sample.kinetic_energy)
        self.assertAlmostEqual(system.grid.integrate(field.density), 2.0, places=8)
        # the pot.dat of the tip run holds the tip's own ionic potential (parsec.F90:2223-2256)
        tip_file = potential_field_from_result(self.result, system)
        np.testing.assert_allclose(
            tip_file.ionic_potential, system.ionic_potential - field.ionic_potential - field.hartree_potential
        )
        self.assertEqual(len(system.point_charges), 2)
        self.assertEqual(tuple(atom.position[2] for atom in system.atoms), (1.9, 3.3))


if __name__ == "__main__":
    unittest.main()
