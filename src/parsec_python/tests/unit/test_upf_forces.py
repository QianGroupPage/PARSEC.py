"""A UPF potential through the solver: forces against finite differences of the energy (any element, no data files).

The synthetic potential of ``test_upf`` has two projectors for s and p and one for d, coupled by a non-diagonal ``D``,
a smooth local potential with a Coulomb tail, and a model core charge.  Two atoms in a small cubic cell, one atom
displaced by +-d: ``-(E(+d) - E(-d)) / 2d`` must be the analytic force, with Andrew Grieder's routines and with the
Fortran-derived ones.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import parsec_python
from parsec_python.tests.unit.test_upf import _upf_text

_SOURCE = str(Path(parsec_python.__file__).resolve().parents[1])

_STEP = 0.02  # bohr
_INPUT = """{routines}Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  7.0 7.0 7.0
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.4
Expansion_Order: 8
Ignore_Symmetry: .true.
Output_Level: 1
Atom_Types_Num: 1
Total_Atom_Num: 2
Atom_Type: Si
Element_Symbol: Si
Pseudopotential_Format: upf
Pseudopot_File: Si.upf
Cubic_Spline: true
begin Atom_Coord
  1.9 2.1 2.3
  {x:.8f} 3.9 3.1
end Atom_Coord
Correlation_Type: pbe
Spin_Polarization: false
States_Num: 8
Net_Charges: 0 e
Fermi_Temp: 1000 K
Max_Iter: 100
Convergence_Criterion: 1e-10 Ry
Mixing_Method: Anderson
Mixing_Param: 0.30
Memory_Param: 4
Diag_Tolerance: 1e-8
Initial_Density: sad
Skip_Force: false
"""


def _run(directory: Path, routines: str, x: float) -> tuple[float, list[list[float]]]:
    directory.mkdir(exist_ok=True)
    (directory / "Si.upf").write_text(_upf_text(element="Si"), encoding="utf-8")
    (directory / "parsec.in").write_text(_INPUT.format(routines=routines, x=x), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "parsec_python", "parsec.in", "--quiet", "--no-archive", "--device", "cpu", "--threads", "1"],
        cwd=directory,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [_SOURCE, os.environ.get("PYTHONPATH")]))},
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr[-2000:])
    text = (directory / "parsec.out").read_text(encoding="utf-8")
    energy = float(re.findall(r"Total Energy =\s+(\S+) \[Ry\]", text)[-1])
    block = text.split("coordinates and symmetrized tot. forces")[1].split("tot.  (net) force")[0]
    forces = [[float(v) for v in line.split()[4:7]] for line in block.splitlines() if re.match(r"\s+\d+\s", line)]
    return energy, forces


class UpfForcesTests(unittest.TestCase):
    def check(self, routines: str) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, forces = _run(root / "base", routines, 4.4)
            plus, _ = _run(root / "plus", routines, 4.4 + _STEP)
            minus, _ = _run(root / "minus", routines, 4.4 - _STEP)
        finite = -(plus - minus) / (2.0 * _STEP)
        analytic = forces[1][0]
        self.assertGreater(abs(analytic), 5.0e-3, "the displacement must give a force to compare")
        # truncation error of the central difference ~ d^2; 5% of the force or 1e-4 Ry/bohr
        self.assertAlmostEqual(analytic, finite, delta=max(1.0e-4, 0.03 * abs(analytic)))

    def test_forces_with_andrew_grieders_routines(self) -> None:
        self.check("Use_Fortran_Routines: .false.\n")

    def test_forces_with_the_fortran_derived_routines(self) -> None:
        self.check("Use_Fortran_Routines: .true.\nPeriodic_Local_Potential: g_space\n")


if __name__ == "__main__":
    unittest.main()
