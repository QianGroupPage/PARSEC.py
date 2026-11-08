"""Wires and slabs: the Gamma-point loop and the one-point Bloch loop are the same calculation.

``run_scf`` (real orbitals) and ``run_scf_kpoints`` at ``k = 0`` must give the same energy; the command line hands a
spin-polarized cell at Gamma to the latter (``cli._gamma_spin_on_kpoint_driver``).  The k-point loop builds its
projectors for :attr:`parsec_python.SCF.pbc.PeriodicPreparedSinglePointSystem.cell_atoms`; on the confined grid of the
Fortran-derived routines that fold moved the atoms by the unit vectors of the confined directions up to 2026-11-08
(an H2 slab 9.4e-4 Ry, an H2 wire 1.5e-4 Ry from the Gamma loop).  It now folds along the periodic vectors only.

The forces agree too.  Up to 2026-11-08 the nonlocal force of a hybrid's k-point result on the Fortran-derived slab
came from eigenvectors of ``input_effective_potential`` without the exchange operator (6.6e-3 Ry/bohr from the Gamma
loop for PBE0); it now reads the orbitals of the Fock loop.  And the nonlocal force the k-point loops accumulate
(``compute_forces``) took projector gradients with images along the placeholder vectors of the confined directions.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Forces import compute_forces
from parsec_python.Forces.wire_slab import wire_slab_kpoint_nonlocal_forces
from parsec_python.Input import parse_parsec_input
from parsec_python.SCF import prepare_periodic_single_point, run_scf
from parsec_python.SCF.hybrid_periodic import run_periodic_hybrid_scf
from parsec_python.SCF.kpoints import run_scf_kpoints

_DATA = Path(__file__).resolve().parents[1] / "data"
_INPUT = """Periodic_System: .true.
Boundary_Conditions: {boundary}
Use_Fortran_Routines: .{fortran}.
begin Cell_Shape
{cell}
end Cell_Shape
Lattice_Vector_Scale: 1
Boundary_Sphere_Radius: 5
Grid_Spacing: 0.55
Expansion_Order: 8
Atom_Types_Num: 1
Atom_Type: H
Element_Symbol: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
  1.0 1.0 0.0
  1.0 1.3 1.4
end Atom_Coord
Correlation_Type: pbe
States_Num: 3
Convergence_Criterion: 1e-9 Ry
Max_Iter: 80
"""
_SLAB_CELL = "  5.0 0.0 0\n  2.0 5.0 0"  # oblique in-plane vectors
_WIRE_CELL = "  4.0"


class ConfinedBlochLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="parsec_confined_bloch_"))
        shutil.copy(_DATA / "H_POTRE.DAT", self.directory / "H_POTRE.DAT")
        self.environment = mock.patch.dict(os.environ, {"PARSEC_KPOINT_BACKEND": "cpu", "PARSEC_KPOINT_WORKERS": "1"})
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        shutil.rmtree(self.directory, ignore_errors=True)

    def _system(self, boundary: str, fortran: str, cell: str, functional: str = "pbe"):
        path = self.directory / f"{boundary}_{fortran}_{functional}.in"
        text = _INPUT.format(boundary=boundary, fortran=fortran, cell=cell)
        path.write_text(text.replace("Correlation_Type: pbe\n", f"Correlation_Type: {functional}\n"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return prepare_periodic_single_point(parse_parsec_input(path).problem)

    def test_the_folded_atoms_keep_their_confined_coordinates(self) -> None:
        for boundary, cell, confined in (("slab", _SLAB_CELL, (2,)), ("wire", _WIRE_CELL, (1, 2))):
            with self.subTest(boundary=boundary):
                system = self._system(boundary, "true", cell)
                self.assertTrue(getattr(system.grid, "is_confined_periodic", False))
                for atom, folded in zip(system.atoms, system.cell_atoms):
                    for axis in confined:
                        self.assertEqual(folded.position[axis], atom.position[axis])

    def test_gamma_loop_is_the_one_point_bloch_loop(self) -> None:
        cases = (
            ("slab", "true", _SLAB_CELL),  # the Fortran-derived slab (ConfinedPeriodicGrid)
            ("slab", "false", _SLAB_CELL),  # Andrew Grieder's slab
            ("wire", "true", _WIRE_CELL),  # the Fortran-derived wire
        )
        for boundary, fortran, cell in cases:
            with self.subTest(boundary=boundary, fortran=fortran):
                system = self._system(boundary, fortran, cell)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gamma = run_scf(system)
                    bloch = run_scf_kpoints(system, np.zeros((1, 3)), np.ones(1))
                self.assertTrue(gamma.converged and bloch.converged)
                self.assertAlmostEqual(gamma.energies.total, bloch.energies.total, delta=1.0e-8)

    def test_hybrid_forces_of_the_gamma_loop_and_the_one_point_bloch_loop(self) -> None:
        # PBE0 on the Fortran-derived slab: the forces of the Bloch result read the orbitals of its Fock loop.
        system = self._system("slab", "true", _SLAB_CELL, functional="pbe0")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gamma = run_scf(system)
            bloch = run_periodic_hybrid_scf(system, np.zeros((1, 3)), np.ones(1), compute_forces=True)
            gamma_forces = compute_forces(system, gamma)
            bloch_forces = compute_forces(system, bloch)
        self.assertTrue(gamma.converged and bloch.converged)
        self.assertIsNotNone(gamma.exact_exchange)
        self.assertIsNotNone(bloch.exact_exchange)
        self.assertAlmostEqual(gamma.energies.total, bloch.energies.total, delta=1.0e-8)
        np.testing.assert_allclose(np.asarray(bloch_forces.total), np.asarray(gamma_forces.total), rtol=0, atol=1.0e-8)
        # the force the loop accumulates (bloch_nonlocal_forces, with the gradients of the confined images) is the
        # nonlocal part of the wire/slab forces
        np.testing.assert_allclose(
            np.asarray(bloch.nonlocal_forces), wire_slab_kpoint_nonlocal_forces(system, bloch), rtol=0, atol=1.0e-12
        )

    def test_the_nonlocal_force_of_the_kpoint_loop_on_a_confined_grid(self) -> None:
        # A k-point off Gamma (in the plane of the slab): the Bloch phases of the projectors and their gradients.
        system = self._system("slab", "true", _SLAB_CELL)
        k_points = np.array([[0.0, 0.0, 0.0], [0.21, -0.13, 0.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_scf_kpoints(system, k_points, np.array([0.5, 0.5]), compute_forces=True)
            # the wire/slab path diagonalizes every k-point again (a semilocal result) to a tight residual
            reference = wire_slab_kpoint_nonlocal_forces(system, result)
        self.assertTrue(result.converged)
        self.assertGreater(float(np.max(np.abs(reference))), 1.0e-2)
        np.testing.assert_allclose(np.asarray(result.nonlocal_forces), reference, rtol=0, atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
