"""Force tables of pseudopotentials shorter than ``forcloc.f90``'s tail offset.

``forcloc.f90:153`` takes the Coulomb tail from ``rlarge = rs(npoint-6)``.
The file samples are ``rs(2:npoint)`` (pseudo.f90:409-410) and the padding
below them is ``rs(j) = j - 1`` (pseudo.f90:843), so a table of fewer than
seven samples puts the tail at zero or below: every point from ``rs(2)``
outwards takes ``2 Z / r**2`` and no density derivative, and points closer
than ``rs(2)`` are skipped first (forcloc.f90:180-188).  The synthetic
``tests/data/H_POTRE.DAT`` has six samples; H2 forces with it used to stop
with an IndexError on both command lines.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python.Pseudopotential.potre import read_parsec_pseudopotential
from parsec_python.Pseudopotential.radial_derivatives import (
    evaluate_density_derivative,
    evaluate_local_potential_derivative,
    tail_start_radius,
)

DATA = Path(__file__).resolve().parents[1] / "data"

H2 = """\
Boundary_Conditions: cluster
Boundary_Sphere_Radius: 5.0 bohr
Grid_Spacing: 0.5 bohr
Coordinate_Unit: Cartesian_Bohr
Ignore_Symmetry: .true.
Hartree_Boundary_Tolerance: off
Expansion_Order: 4
States_Num: 4
Fermi_Temp: 500.0
Max_Iter: 60
Convergence_Criterion: 1.0e-8
Eigensolver: chebff
FF_Maxiter: 9
Chebdav_Degree: 40
Mixing_Param: 0.3
Skip_Force: .false.
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
   0.0 0.0 -0.7
   0.0 0.0 0.7
end Atom_Coord
Correlation_Type: ca
"""


def _forces(text: str) -> np.ndarray:
    """The last total force table of parsec.out (Ry/bohr), one row per atom."""

    block = text.split("coordinates and tot. forces (after setting net force to zero)")[-1]
    block = block.split("tot. torque")[0]
    rows = re.findall(
        r"^\s+\d+\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s*$",
        block,
        re.M,
    )
    return np.asarray([[float(value) for value in row[3:]] for row in rows])


class ShortRadialTableTests(unittest.TestCase):
    def test_tail_radius_follows_the_fortran_padding(self) -> None:
        short = read_parsec_pseudopotential(DATA / "H_POTRE.DAT")
        self.assertEqual(short.radii.size, 6)
        # npoint = 7: rs(npoint-6) = rs(1) = 0.
        self.assertEqual(tail_start_radius(short), 0.0)
        # Five samples: rs(0) = -1; seven: the first file sample rs(2).
        self.assertEqual(tail_start_radius(SimpleNamespace(radii=np.arange(1.0, 6.0))), -1.0)
        self.assertEqual(tail_start_radius(SimpleNamespace(radii=np.arange(1.0, 8.0))), 1.0)
        normal = read_parsec_pseudopotential(DATA / "water_H_POTRE.DAT")
        self.assertEqual(tail_start_radius(normal), float(normal.radii[-7]))

    def test_short_table_takes_the_coulomb_tail_beyond_the_first_sample(self) -> None:
        short = read_parsec_pseudopotential(DATA / "H_POTRE.DAT")
        distances = np.array([0.05, 0.1, 1.0, 5.0])
        values = evaluate_local_potential_derivative(short, distances, 0)
        expected = np.where(distances >= 0.1, 2.0 * short.ionic_charge / distances**2, 0.0)
        np.testing.assert_array_equal(values, expected)
        density = evaluate_density_derivative(short, distances, short.valence_density)
        np.testing.assert_array_equal(density, np.zeros(4))

    def test_normal_tables_keep_their_values(self) -> None:
        normal = read_parsec_pseudopotential(DATA / "water_H_POTRE.DAT")
        distances = np.concatenate(([0.5 * normal.radii[0]], np.geomspace(normal.radii[0], 2.0 * normal.radii[-1], 400)))
        radius = distances
        tail = radius >= normal.radii[-7]
        values = evaluate_local_potential_derivative(normal, distances, 0)
        np.testing.assert_array_equal(values[tail], 2.0 * normal.ionic_charge / radius[tail] ** 2)
        self.assertEqual(values[0], 0.0)

    def test_h2_forces_with_the_short_table_on_both_command_lines(self) -> None:
        from parsec_python.cli import main as reference_main

        tables = []
        for main in ("reference", "accelerated"):
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
                directory = Path(folder)
                shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
                (directory / "parsec.in").write_text(H2, encoding="utf-8")
                if main == "reference":
                    run = reference_main
                else:
                    from parsec_python.acceleration.cli import main as run
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as error:
                    code = run([str(directory / "parsec.in"), "--quiet", "--no-archive"])
                self.assertEqual(code, 0, error.getvalue())
                tables.append(_forces((directory / "parsec.out").read_text()))
        reference, accelerated = tables
        self.assertEqual(reference.shape, (2, 3))
        self.assertGreater(float(np.abs(reference).max()), 0.0)
        np.testing.assert_allclose(accelerated, reference, atol=2.0e-6)
        # The two atoms feel opposite forces along the bond.
        np.testing.assert_allclose(reference[0], -reference[1], atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
