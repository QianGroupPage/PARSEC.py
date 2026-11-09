"""Shifted Monkhorst-Pack k-points folded into (-1/2, 1/2] (a decision of this port, PARSEC 1.5's intent).

PARSEC 1.5 keeps the points of ``Monkhorst_Pack_Grid`` + ``Monkhorst_Pack_Shift`` in [0, 1) of the reciprocal
lattice (``irrbz.f90``: the swap to the shortest member never runs, ``norm2`` carries ``+ eight``).  The
finite-difference Bloch operator ``-nabla^2 - 2i k.grad + k^2`` is not invariant under ``k -> k + G``, so a run
without symmetry, which computes ``3/4 b`` itself, differed from the symmetric run, which reaches ``-1/4 b`` by
rotating its representative (Si8 diamond, MP 2x2x2 shifted, h 0.64: 15 mRy).  Both sets of routines fold the
points; explicit k-point lists stay as written.
"""

# With both sets of routines: the energy test runs the Fortran-derived routines and Andrew Grieder's.
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Grid.kpoints import fold_to_centered_zone, irreducible_monkhorst_pack
from parsec_python.Input.parsec_input import parse_parsec_input

_DATA = Path(__file__).resolve().parents[1] / "data"

# one Si atom in a simple cubic cell: the 48 operations map the shifted grid, and the 2x2x2 shifted mesh has the
# points 1/4 and 3/4 along every direction
_SI_SC = """Periodic_System: .true.
Boundary_Conditions: bulk
Use_Fortran_Routines: {routines}
begin Cell_Shape
 5.0 5.0 5.0
end Cell_Shape
Coordinate_Unit: lattice_vectors
Grid_Spacing: 0.5
Expansion_Order: 6
{extra}
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
 2 2 2
end Monkhorst_Pack_Grid
begin Monkhorst_Pack_Shift
 0.5 0.5 0.5
end Monkhorst_Pack_Shift
States_Num: 10
Fermi_Temp: 1000.0
Max_Iter: 60
Convergence_Criterion: 1.0e-7
Mixing_Method: Anderson
Mixing_Param: 0.3
Correlation_Type: ca
Atom_Types_Num: 1
Atom_Type: Si
Local_Component: p
begin Atom_Coord
 0.0 0.0 0.0
end Atom_Coord
"""

_PLAIN = "Ignore_Symmetry: .true.\nbegin Grid_Shift\n0.5 0.5 0.5\nend Grid_Shift"


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)
        shutil.copy(_DATA / "Si_POTRE.DAT", self.directory / "Si_POTRE.DAT")

    def write(self, name: str, text: str) -> Path:
        folder = self.directory / name
        folder.mkdir()
        shutil.copy(self.directory / "Si_POTRE.DAT", folder / "Si_POTRE.DAT")
        path = folder / "parsec.in"
        path.write_text(text, encoding="utf-8")
        return path


class FoldingConventionTests(unittest.TestCase):
    def test_the_half_stays_and_folding_is_idempotent(self) -> None:
        values = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0, -0.5, -0.25, 1.5, 0.49999999999999994])
        folded = fold_to_centered_zone(values)
        np.testing.assert_array_equal(folded, [0.0, 0.25, 0.5, -0.25, 0.9 - 1.0, 0.0, 0.5, -0.25, 0.5, values[-1]])
        np.testing.assert_array_equal(fold_to_centered_zone(folded), folded)
        # components in [0, 1/2] keep their bits (a grid whose points all lie there is unchanged)
        quarter = np.array([0.0, 0.1, 0.25, 0.5])
        self.assertEqual(fold_to_centered_zone(quarter).tobytes(), quarter.tobytes())
        self.assertTrue(np.all((folded > -0.5) & (folded <= 0.5)))

    def test_the_reduced_wedge_keeps_its_counts_and_weights(self) -> None:
        lattice = np.diag([10.26, 10.26, 10.26])
        folded = irreducible_monkhorst_pack(lattice, (2, 2, 2), (0.5, 0.5, 0.5), None)
        plain = irreducible_monkhorst_pack(lattice, (2, 2, 2), (0.5, 0.5, 0.5), None, fold=False)
        np.testing.assert_array_equal(plain.fractional[:, 2], [0.25, 0.75, 0.25, 0.75])
        np.testing.assert_array_equal(folded.fractional[:, 2], [0.25, -0.25, 0.25, -0.25])
        for name in ("counts", "weights", "kinetic_energies", "grid_map"):
            np.testing.assert_array_equal(getattr(folded, name), getattr(plain, name))
        # an unshifted 2x2x2 grid has the points 0 and 1/2 only: the same bits as before
        gamma = irreducible_monkhorst_pack(lattice, (2, 2, 2), (0.0, 0.0, 0.0), None)
        gamma_plain = irreducible_monkhorst_pack(lattice, (2, 2, 2), (0.0, 0.0, 0.0), None, fold=False)
        self.assertEqual(gamma.k_points.tobytes(), gamma_plain.k_points.tobytes())


class ParsedKPointTests(_Case):
    def test_both_routine_sets_fold_and_explicit_lists_stay(self) -> None:
        from parsec_python.Grid.kpoints import kpoint_sampling

        reciprocal = 2.0 * np.pi / 5.0
        his = parse_parsec_input(self.write("his", _SI_SC.format(routines=".false.", extra=_PLAIN))).problem
        # since 2026-11-08 Andrew Grieder's routines keep the mesh (reduced by time reversal alone here, with
        # Ignore_Symmetry): its points folded as well
        k_points, _ = kpoint_sampling(his)
        fractions = np.asarray(k_points) / reciprocal
        self.assertEqual(sorted(set(np.round(fractions.ravel(), 12))), [-0.25, 0.25])
        listed = _SI_SC.format(routines=".false.", extra=_PLAIN).replace(
            "Kpoint_Method: mp", "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\n"
            "begin Kpoint_List\n 0.25 0.25 0.75 1.0\nend Kpoint_List"
        )
        listed = listed.split("begin Monkhorst_Pack_Grid")[0] + listed.split("end Monkhorst_Pack_Shift")[1]
        explicit = parse_parsec_input(self.write("listed", listed)).problem
        np.testing.assert_allclose(np.asarray(explicit.kpoints) / reciprocal, [[0.25, 0.25, 0.75]], atol=1e-15)


class SymmetricAndPlainRunsAgreeTests(_Case):
    def energy(self, name: str, text: str) -> float:
        from parsec_python.cli import main as cli_main

        path = self.write(name, text)
        self.assertEqual(cli_main([str(path), "--quiet"]), 0)
        with np.load(path.parent / "parsec_python_results.npz") as archive:
            return float(archive["energy_total_ry"])

    def test_the_symmetric_run_equals_the_run_without_symmetry(self) -> None:
        for routines in (".true.", ".false."):
            with self.subTest(routines=routines):
                symmetric = self.energy("sym" + routines.strip("."), _SI_SC.format(routines=routines, extra=""))
                plain = self.energy("plain" + routines.strip("."), _SI_SC.format(routines=routines, extra=_PLAIN))
                # unfolded, the 3/4 points of the plain run gave a different energy
                self.assertAlmostEqual(symmetric, plain, delta=1.0e-7)


if __name__ == "__main__":
    unittest.main()
