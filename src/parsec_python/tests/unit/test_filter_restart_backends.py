"""``Filter_Restart_After`` with the warm start of every ionic backend (PARSEC 1.5, parsec.F90:2944-3026).

The restart that keeps the potential and drops only the eigensolver subspace
(``Restart_Cheb_Remained_Times`` > ``Remained_Times_Atomic_Chrg_Den``) has to
handle what each backend hands to its next SCF loop
(:mod:`parsec_python.Relaxation.backend`):

* ``SCFWarmStart`` (reference, Gamma and k-point cells, Andrew Grieder's
  backends, the accelerated driver) with one subspace, a spin pair or a
  ``{k-point: state}`` dict;
* the ``_MovedStructureStart`` wrapper of ``fortran_spinor_backend``;
* the bare channel potentials of the spin k-point loop of
  ``fortran_periodic_backend`` (also the wire/slab backend), which carries no
  subspace.

Function tests for each type, the ionic loops (relaxation and molecular
dynamics) with a stand-in backend of each kind, and two command-line runs of
inputs (a spin k-point cell, the AuH spin-orbit molecule, both on the
Fortran-derived routines) that crashed with ``TypeError`` before.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python.Relaxation.backend import (
    IonicBackend,
    SpinorWarmStart,
    _MovedStructureStart,
    _drop_moved_structure_subspace,
    fortran_spinor_backend,
)
from parsec_python.SCF import SCFWarmStart
from parsec_python.SCF.filter_restart import FilterRestart, drop_eigensolver_subspace

AUH = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "reduced_self_consistent_soc"


def _start(state="S", degree=None):
    return SCFWarmStart(
        input_potential=np.arange(3.0),
        hartree_potential=np.ones(3),
        eigensolver_state=state,
        first_iteration_degree=degree,
    )


class DropSubspaceTests(unittest.TestCase):
    def test_every_warm_start_type(self) -> None:
        self.assertIsNone(drop_eigensolver_subspace(None))
        for state, cleared in (("S", None), (("U", "D"), (None, None)), ({(0.0, 0.0, 0.0): "K"}, None)):
            with self.subTest(state=state):
                dropped = drop_eigensolver_subspace(_start(state, degree=40))
                self.assertEqual(dropped.eigensolver_state, cleared)
                self.assertIsNone(dropped.first_iteration_degree)
                np.testing.assert_array_equal(dropped.input_potential, np.arange(3.0))
        spinor = drop_eigensolver_subspace(SpinorWarmStart(potential="V", eigensolver_state="S"))
        self.assertEqual((spinor.potential, spinor.eigensolver_state), ("V", None))
        # the spin k-point loop of the Fortran-derived routines: potentials only
        potentials = (np.zeros(3), np.ones(3))
        self.assertIs(drop_eigensolver_subspace(potentials), potentials)
        wrapped = _drop_moved_structure_subspace(_MovedStructureStart(_start({(0.0, 0.0, 0.0): "K"})))
        self.assertIsInstance(wrapped, _MovedStructureStart)
        self.assertIsNone(wrapped.start.eigensolver_state)
        self.assertIs(fortran_spinor_backend().drop_subspace, _drop_moved_structure_subspace)
        with self.assertRaisesRegex(TypeError, "drop_subspace"):
            drop_eigensolver_subspace(object())

    def test_the_backend_hook_is_used(self) -> None:
        restart = FilterRestart(after=1, restarts_left=1, densities_left=0)
        kept, reset = restart.warm_start_after_move("carried", 1, drop_subspace=lambda carried: ("dropped", carried))
        self.assertEqual((kept, reset), (("dropped", "carried"), "eigensolver"))


class _Backend:
    """A stand-in backend whose warm starts have the type of one of the real backends."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.received: list[object] = []

    def run_scf(self, system, callback=None, warm_start=None, **_):
        self.received.append(warm_start)
        positions = np.array([atom.position for atom in system.atoms], dtype=float)
        energy = float(np.sum((positions - 0.1) ** 2))
        return SimpleNamespace(
            energies=SimpleNamespace(total=energy), converged=True, iterations=1, positions=positions
        )

    def compute_forces(self, system, result, *, symmetrize=True, operations=None):
        return SimpleNamespace(total=-2.0 * (result.positions - 0.1), maximum=1.0)

    def update_geometry(self, system, atoms):
        return SimpleNamespace(atoms=tuple(atoms), input=system.input)

    def warm_start(self, previous_system, previous_result, system):
        if self.kind == "potentials":
            return (np.zeros(3), np.ones(3))
        start = _start({(0.0, 0.0, 0.0): "K"} if self.kind == "spinor" else ("U", "D"), degree=40)
        return _MovedStructureStart(start) if self.kind == "spinor" else start

    def backend(self) -> IonicBackend:
        return IonicBackend(
            run_scf=self.run_scf,
            compute_forces=self.compute_forces,
            update_geometry=self.update_geometry,
            warm_start=self.warm_start,
            drop_subspace=_drop_moved_structure_subspace if self.kind == "spinor" else None,
        )


def _system(filter_restart=(1, 2, 0)):
    from parsec_python.models import Atom

    atoms = (Atom("H", np.array([0.0, 0.0, 0.7])), Atom("H", np.array([0.0, 0.2, -0.7])))
    return SimpleNamespace(atoms=atoms, input=SimpleNamespace(eigensolver=SimpleNamespace(filter_restart=filter_restart)))


def _check(test, kind, received) -> None:
    # Structure 0 starts cold; structures 1 and 2 follow the two eigensolver restarts; 3 is a plain warm start.
    test.assertIsNone(received[0])
    for start in received[1:3]:
        if kind == "potentials":
            test.assertIsInstance(start, tuple)
        elif kind == "spinor":
            test.assertIsInstance(start, _MovedStructureStart)
            test.assertIsNone(start.start.eigensolver_state)
        else:
            test.assertEqual(start.eigensolver_state, (None, None))
            test.assertIsNone(start.first_iteration_degree)
    if len(received) > 3:
        last = received[3]
        if kind == "spinor":
            test.assertIsNotNone(last.start.eigensolver_state)
        elif kind == "scf":
            test.assertEqual(last.eigensolver_state, ("U", "D"))


class IonicLoopTests(unittest.TestCase):
    def test_relaxation_with_each_warm_start(self) -> None:
        from parsec_python.Relaxation import run_relaxation
        from parsec_python.models import RelaxationSettings

        for kind in ("scf", "potentials", "spinor"):
            with self.subTest(kind=kind):
                fake = _Backend(kind)
                settings = RelaxationSettings(method="simple", max_movements=3, minimum_force=1.0e-5)
                run_relaxation(_system(), settings, fake.backend(), symmetrize=False)
                self.assertEqual(len(fake.received), 4)
                _check(self, kind, fake.received)

    def test_molecular_dynamics_with_each_warm_start(self) -> None:
        from parsec_python.MolecularDynamics import MolecularDynamicsSettings
        from parsec_python.MolecularDynamics.run import run_recorded_molecular_dynamics

        settings = MolecularDynamicsSettings(
            enabled=True, step_num=3, use_test_rng=True, rng_seed=5, initial_temperature=300.0, final_temperature=300.0
        )
        eigensolver = SimpleNamespace(filter_restart=(1, 2, 0))
        translation = SimpleNamespace(
            molecular_dynamics=settings,
            movable=(True, True),
            potential_extrapolation="none",
            problem=SimpleNamespace(
                eigensolver=eigensolver,
                pseudopotentials={"H": SimpleNamespace(element_symbol="H")},
            ),
        )
        for kind in ("scf", "potentials", "spinor"):
            with self.subTest(kind=kind):
                fake = _Backend(kind)
                reporter = SimpleNamespace(iteration=lambda *_: None, finish=lambda *_, **__: None)
                run_recorded_molecular_dynamics(
                    translation, _system(), fake.backend(), reporter, lambda text: None, directory=None
                )
                self.assertEqual(len(fake.received), 4)
                _check(self, kind, fake.received)


# Neutral H2: both spin channels stay occupied. The one-electron H2+ of the
# first version flipped its moment for dozens of iterations and converged only
# on some grids (13^3 and 16^3; PARSEC 1.5's adjust_step gives 15^3 here).
_SPIN_CELL = """
Periodic_System: .true.
Boundary_Conditions bulk
begin Cell_Shape
 6.0 0.0 0.0
 0.0 6.0 0.0
 0.0 0.0 6.0
end Cell_Shape
Lattice_Vector_Scale: 1.0
Grid_Spacing: 0.4
Expansion_Order: 8
Correlation_Type: ca
States_Num: 3
Convergence_Criterion: 1e-5 Ry
Max_Iter: 80
Eigensolver: chebdav
Fermi_Temp: 300 K
Mixing_Param: 0.3
Net_Charges: 0 e
Spin_Polarization: .true.
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
2 1 1
end Monkhorst_Pack_Grid
Use_Fortran_Routines: .true.
Minimization: bfgs
Movement_Num: 2
Force_Min: 1e-4 Ry/bohr
{filter}
Atom_Types_Num: 1
Coordinate_Unit: cartesian_bohr
Atom_Type: H
Element_Symbol: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
 2.9 3.1 3.0
 4.35 3.0 3.2
end Atom_Coord
"""

_FILTER = "Filter_Restart_After: 1\nRestart_Cheb_Remained_Times: 2\nRemained_Times_Atomic_Chrg_Den: 0\n"

_H2_CLUSTER = """Boundary_Conditions: cluster
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.4
Coordinate_Unit: Cartesian_Bohr
Hartree_Boundary_Tolerance: off
States_Num: 3
Convergence_Criterion: 1e-7 Ry
Eigensolver: chebdav
Minimization: bfgs
Movement_Num: 2
Force_Min: 1e-4 Ry/bohr
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
 0.0 0.0 0.80
 0.0 0.0 -0.78
end Atom_Coord
"""


def _run_cli(main, path: Path) -> tuple[int, str]:
    output, error = StringIO(), StringIO()
    with redirect_stdout(output), redirect_stderr(error):
        code = main([str(path), "--quiet", "--no-archive"])
    return code, error.getvalue()


class CommandLineTests(unittest.TestCase):
    """The two inputs that stopped with ``TypeError`` after the first move."""

    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(scratch.cleanup)
        self.directory = Path(scratch.name)

    def test_spin_kpoint_cell_of_the_fortran_routines(self) -> None:
        from parsec_python.cli import main

        shutil.copy(AUH.parent / "H_POTRE.DAT", self.directory / "H_POTRE.DAT")
        path = self.directory / "parsec.in"
        path.write_text(_SPIN_CELL.format(filter=_FILTER), encoding="utf-8")
        code, error = _run_cli(main, path)
        self.assertIn(code, (0, 4), error)
        text = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertNotIn("Calculation failed", text + error)
        # three structures: the first, and two after eigensolver restarts
        self.assertEqual(text.count("Time for movement"), 3)

    def test_accelerated_cluster_restarts_its_device_subspace(self) -> None:
        """The accelerated driver's warm start (device subspaces): a ``None`` state resets them.

        The restarted structures reach the energies of the run without the
        restart (only the start vectors differ).
        """

        import re

        from parsec_python.acceleration.cli import main

        shutil.copy(AUH.parent / "H_POTRE.DAT", self.directory / "H_POTRE.DAT")
        energies = {}
        for label, extra in (("plain", ""), ("restarted", _FILTER)):
            path = self.directory / "parsec.in"
            path.write_text(_H2_CLUSTER + extra, encoding="utf-8")
            code, error = _run_cli(main, path)
            self.assertIn(code, (0, 4), error)
            text = (self.directory / "parsec.out").read_text(encoding="utf-8")
            self.assertNotIn("Calculation failed", text + error)
            energies[label] = [float(value) for value in re.findall(r"Total Energy =\s+(-?\d+\.\d+) \[Ry\]", text)]
        self.assertGreaterEqual(len(energies["restarted"]), 3)
        np.testing.assert_allclose(energies["restarted"][-1], energies["plain"][-1], atol=1.0e-5)

    def test_spin_orbit_molecule_of_the_fortran_routines(self) -> None:
        from parsec_python.cli import main

        for name in ("Au_POTRE.DAT", "H_POTRE.DAT"):
            shutil.copy(AUH / name, self.directory / name)
        base = (AUH / "parsec.in").read_text(encoding="utf-8")
        path = self.directory / "parsec.in"
        path.write_text(
            base
            + "\nUse_Fortran_Routines: .true.\nMinimization: bfgs\nMovement_Num: 1\nForce_Min: 0.001\n"
            + "Filter_Restart_After: 1\nRestart_Cheb_Remained_Times: 1\nRemained_Times_Atomic_Chrg_Den: 0\n",
            encoding="utf-8",
        )
        code, error = _run_cli(main, path)
        self.assertIn(code, (0, 4), error)
        text = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertNotIn("Calculation failed", text + error)
        self.assertEqual(text.count("Time for movement"), 2)


if __name__ == "__main__":
    unittest.main()
