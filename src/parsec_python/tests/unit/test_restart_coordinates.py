"""Restart_Run after a first run that centred the cluster, against Fortran PARSEC.

``initial.f90:133-134`` centres a cluster only in a first run (``istart = 0``)
and ``wfnsave.F90`` stores no coordinates in parsec.dat: PARSEC restarts the
stored potential at the input coordinates.  For an H2O input whose bounding
box is off the origin by 0.4034 bohr (along y) that potential belongs to the
centred atoms, and the restart converges to another fixed point.  The WSL
binary (one rank, Ignore_Symmetry, the potentials of tests/data) gives
-32.76819114 Ry for the first run and -33.34358015 Ry for the restart
(a water run); this port follows the same rules and reaches the
same two energies (``acceleration/tests/test_geigen_parity.py`` checks the
accelerated command line).
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"

#: Fortran PARSEC, printed to 1e-8 Ry: first run, restart.
FORTRAN_ENERGIES = (-32.76819114, -33.34358015)
BOHR_PER_ANGSTROM = 1.0 / 0.529177  # PARSEC's angs (const.f90:40), the parser's factor

H2O = """\
Restart_Run: {restart}
Boundary_Conditions: cluster
Cluster_Domain_Shape: sphere
Boundary_Sphere_Radius: 10.4 bohr
Grid_Spacing: 0.57 bohr
Coordinate_Unit: Cartesian_Ang
Expansion_Order: 8
Hartree_Boundary_Tolerance: off
Solver_Lpole: 9
States_Num: 8
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 60
Convergence_Criterion: 1.0e-5
Diag_Tolerance: 1.0e-6
Eigensolver: chebff
Chebdav_Degree: 20
FF_Maxiter: 2
Chebyshev_Degree: 15
Chebyshev_Degree_Delta: 3
Mixing_Method: Anderson
Mixing_Param: 0.3
Atom_Types_Num: 2
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
   0.000000  -0.488456   0.950000
   0.000000  -0.488456  -0.950000
end Atom_Coord
Atom_Type: O
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
   0.000000   0.061544   0.000000
end Atom_Coord
Correlation_Type: ca
Ignore_Symmetry: .true.
"""

INPUT_Y = np.array([-0.488456, -0.488456, 0.061544]) * BOHR_PER_ANGSTROM


def first_and_restart(test: unittest.TestCase, main) -> list[dict]:
    """Run the first calculation and its restart with ``main``; their archives."""

    records = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        directory = Path(folder)
        shutil.copy(DATA / "water_H_POTRE.DAT", directory / "H_POTRE.DAT")
        shutil.copy(DATA / "water_O_POTRE.DAT", directory / "O_POTRE.DAT")
        for restart in (".false.", ".true."):
            (directory / "parsec.in").write_text(H2O.format(restart=restart), encoding="utf-8")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as error:
                code = main([str(directory / "parsec.in"), "--quiet"])
            test.assertEqual(code, 0, error.getvalue())
            with np.load(directory / "parsec_python_results.npz") as archive:
                records.append(
                    dict(
                        energy=float(archive["energy_total_ry"]),
                        coordinates=np.asarray(archive["atom_coordinates_bohr"]),
                        log=(directory / "parsec.out").read_text(),
                    )
                )
    return records


def check_against_fortran(test: unittest.TestCase, records: list[dict]) -> None:
    first, restart = records
    # The first run centres the bounding box (initial.f90:158-183) ...
    shift = 0.5 * (INPUT_Y.min() + INPUT_Y.max())
    np.testing.assert_allclose(first["coordinates"][:, 1], INPUT_Y - shift, atol=1.0e-9)
    # ... the restart keeps the input coordinates (istart = 1).
    np.testing.assert_allclose(restart["coordinates"][:, 1], INPUT_Y, atol=1.0e-9)
    test.assertIn("Restart_Run: the atom coordinates are used as the input gives them", restart["log"])
    test.assertIn("Restarting from previous run", restart["log"])
    for record, fortran in zip(records, FORTRAN_ENERGIES):
        test.assertLess(abs(record["energy"] - fortran), 1.0e-8)


class RestartCoordinateTests(unittest.TestCase):
    def test_reference_restart_reaches_the_fortran_energies(self) -> None:
        from parsec_python.cli import main

        check_against_fortran(self, first_and_restart(self, main))


if __name__ == "__main__":
    unittest.main()
