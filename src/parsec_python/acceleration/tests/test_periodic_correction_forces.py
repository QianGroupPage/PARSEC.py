"""Forces, relaxation and dynamics of bulk cells with LDA+U or TS on the accelerated Gamma path; LDA+U bands.

The accelerated forces of a periodic cell (``Forces/periodic_device.py``) take the LDA+U term from
the orbitals where the eigensolver keeps them, with the projector gradients of the cell images
(``Hamiltonian/hubbard_device.py``), and the TS forces of the device sums.  Checked against the
reference periodic backend: forces, finite differences (of ``E - T S`` with LDA+U; with TS of
``E - E_vdW`` against the forces without the TS term and of ``E_vdW`` at fixed Hirshfeld volumes
against the TS forces, since efvdw.f90 leaves the derivative of the volumes out), the trajectories
of the two command lines, the routes, and the LDA+U bands of the CPU and the CuPy k-point workers.
"""


# With both sets of routines: the periodic inputs and grids of this module are those of the Fortran-derived
# routines (Use_Fortran_Routines: .true., PeriodicGridSettings(fortran_routines=True)); the numbers checked
# here are those of the Fortran parity round.
from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Input import parse_parsec_input
from parsec_python.models import Atom
from parsec_python.acceleration import driver
from parsec_python.acceleration.backends.cupy import cupy_available

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

_COMMON = """Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
Grid_Spacing: 0.6
Expansion_Order: 6
Ignore_Symmetry: .true.
Fermi_Temp: 300.0
Max_Iter: 100
Convergence_Criterion: 1.0e-9
Diag_Tolerance: 1.0e-9
Mixing_Method: Anderson
Mixing_Param: 0.3
Eigensolver: chebff
"""

# Two Ag atoms closer through the cell boundary than inside it.
_AG = """begin Cell_Shape
   6.0  6.0  6.0
end Cell_Shape
States_Num: 16
Correlation_Type: ca
{extra}
Atom_Types_Num: 1
Atom_Type: Ag
Local_Component: s
ldaplusu_u 4.0 eV
begin Atom_Coord
   0.8 2.9 3.1
   3.9 3.2 2.8
end Atom_Coord
"""

# Two H2 in a small cell (PARSEC's TS correction is that of PBE); the atoms stay away from the
# faces of the cell centred on the origin, where the Hirshfeld weights of Hf_Three = 0 jump.
_H4 = """begin Cell_Shape
   7.2  7.2  8.4
end Cell_Shape
Van_Der_Waals: .true.
States_Num: 6
Correlation_Type: pbe
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  3.0 3.1 1.8
  3.0 3.1 3.2
  3.3 3.3 5.5
  3.3 3.3 6.9
end Atom_Coord
"""

_ROOT: tempfile.TemporaryDirectory | None = None


def setUpModule():
    global _ROOT
    _ROOT = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def tearDownModule():
    if _ROOT is not None:
        _ROOT.cleanup()


def _write(directory: Path, name: str, extra: str = "") -> Path:
    if name == "ag":
        text = _COMMON + _AG.format(extra=extra)
        shutil.copy(DATA / "Ag_POTRE.DAT", directory / "Ag_POTRE.DAT")
    else:
        text = _COMMON + _H4.format(extra=extra)
        shutil.copy(DATA / "spin_pbe_H_POTRE.DAT", directory / "H_POTRE.DAT")
    path = directory / "parsec.in"
    path.write_text(text, encoding="utf-8")
    return path


def _translation(name: str, extra: str = ""):
    return parse_parsec_input(_write(Path(tempfile.mkdtemp(dir=_ROOT.name)), name, extra))


def _free_energy(result, temperature=300.0):
    """``E - T S`` (the free energy the forces differentiate)."""

    occupations = np.clip(np.asarray(result.occupations, dtype=float), 1e-300, 1 - 1e-16)
    entropy = occupations * np.log(occupations) + (1 - occupations) * np.log(1 - occupations)
    return float(result.energies.total + 6.33327186e-6 * temperature * 2.0 * np.sum(entropy))


def _moved(atoms, index, axis, step):
    out = list(atoms)
    position = np.array(out[index].position, dtype=float)
    position[axis] += step
    out[index] = Atom(out[index].symbol, position)
    return tuple(out)


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class GammaForceTest(unittest.TestCase):
    """The accelerated Gamma path (orbitals kept on the device) against the reference backend."""

    def run_both(self, problem):
        from parsec_python.Relaxation.backend import fortran_periodic_backend as periodic_backend
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        from ..Forces import periodic_device as device_forces

        reference_backend = periodic_backend()
        reference_system = prepare_periodic_single_point(problem)
        reference_result = reference_backend.run_scf(reference_system)
        reference = reference_backend.compute_forces(
            reference_system, reference_result, symmetrize=False
        )
        scope = device_forces.device_periodic_forces()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        system = driver.prepare_single_point(problem, backend="cupy", symmetry="off")
        # What an ionic loop of the command line does: the orbitals stay on the device.
        system.materialize_final_wavefunctions = False
        backend = device_forces.with_periodic_operations(driver.ionic_backend())
        result = backend.run_scf(system)
        forces = backend.compute_forces(system, result, symmetrize=False)
        return reference, system, backend, result, forces

    def test_lda_plus_u_forces_equal_the_reference_and_the_free_energy_slope(self):
        problem = _translation("ag").problem
        reference, system, backend, _result, forces = self.run_both(problem)
        self.assertGreater(np.max(np.abs(reference.hubbard)), 1.0e-3)
        np.testing.assert_allclose(forces.hubbard, reference.hubbard, rtol=0.0, atol=1.0e-10)
        np.testing.assert_allclose(forces.unsymmetrized, reference.unsymmetrized, rtol=0.0, atol=1.0e-8)
        step = 1.0e-5
        energies = [
            _free_energy(backend.run_scf(backend.update_geometry(system, _moved(problem.atoms, 1, 0, sign * step))))
            for sign in (1.0, -1.0)
        ]
        numeric = -(energies[0] - energies[1]) / (2.0 * step)
        self.assertAlmostEqual(forces.unsymmetrized[1, 0], numeric, delta=1.0e-6)

    def test_ts_forces_equal_the_reference_and_split_finite_differences(self):
        from parsec_python.Energy.dispersion import species_elements

        from ..Energy.dispersion_device import device_ts_dispersion

        problem = _translation("h4").problem
        reference, system, backend, result, forces = self.run_both(problem)
        np.testing.assert_allclose(forces.dispersion, reference.dispersion, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(forces.unsymmetrized, reference.unsymmetrized, rtol=0.0, atol=1.0e-8)
        lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=float)
        elements = species_elements(system.atoms, problem.pseudopotentials)
        step = 1.0e-5
        index, axis = 0, 2
        rest, pair = [], []
        for sign in (1.0, -1.0):
            moved = backend.update_geometry(system, _moved(problem.atoms, index, axis, sign * step))
            moved_result = backend.run_scf(moved)
            rest.append(_free_energy(moved_result) - moved_result.dispersion.energy)
            pair.append(device_ts_dispersion(moved.atoms, elements, result.dispersion.volumes, lattice).energy)
        without = forces.unsymmetrized[index, axis] - result.dispersion.forces[index, axis]
        self.assertAlmostEqual(without, -(rest[0] - rest[1]) / (2.0 * step), delta=1.0e-6)
        self.assertAlmostEqual(
            result.dispersion.forces[index, axis], -(pair[0] - pair[1]) / (2.0 * step), delta=1.0e-9
        )


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class CommandLineTest(unittest.TestCase):
    """The accelerated command line runs them itself, on the reference solver's trajectory."""

    def run_both(self, name: str, extra: str):
        from parsec_python.cli import main as reference_main

        from ..cli import main as accelerated_main

        archives, messages = [], []
        for label, main in (("accelerated", accelerated_main), ("reference", reference_main)):
            folder = Path(tempfile.mkdtemp(dir=_ROOT.name)) / label
            folder.mkdir()
            path = _write(folder, name, extra)
            errors = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(errors):
                code = main([str(path), "--quiet"])
            self.assertIn(code, (0, 4), errors.getvalue())
            messages.append(errors.getvalue())
            with np.load(folder / "parsec_python_results.npz", allow_pickle=True) as archive:
                archives.append({key: archive[key] for key in archive.files})
        return archives, messages

    def test_lda_plus_u_relaxation_follows_the_reference(self):
        (accelerated, reference), (message, _) = self.run_both(
            "ag", "Minimization: bfgs\nMovement_Num: 2\nForce_Min: 0.0005\n"
        )
        self.assertNotIn("reference parsec_python.cli solver", message)
        self.assertEqual(str(accelerated["backend_selected"]), "cupy")
        for key in ("relaxation_positions", "relaxation_energies", "relaxation_forces"):
            np.testing.assert_allclose(accelerated[key], reference[key], rtol=0, atol=1e-8, err_msg=key)

    def test_ts_dynamics_follows_the_reference(self):
        (accelerated, reference), (message, _) = self.run_both(
            "h4",
            "Molecular_Dynamics: .true.\nStep_num: 2\nTinit: 300.0\nT_Final: 300.0\n"
            "T_Step: 0.0\nTime_Step: 20.0\ntest_rng: .true.\n",
        )
        self.assertNotIn("reference parsec_python.cli solver", message)
        keys = [key for key in reference if key.startswith("dynamics_") and reference[key].dtype.kind == "f"]
        self.assertTrue(keys)
        for key in keys:
            np.testing.assert_allclose(accelerated[key], reference[key], rtol=0, atol=1e-7, err_msg=key)


class ShellTest(unittest.TestCase):
    def test_shell_steps_follow_the_order_of_the_loops(self):
        # The host and device pair sums add the terms of a shell in this order.
        from parsec_python.Energy.dispersion import lattice_shell_steps

        for n in range(1, 9):
            loops = np.array(
                [
                    (i, j, k)
                    for i in range(-n, n + 1)
                    for j in range(-n, n + 1)
                    for k in range(-n, n + 1)
                    if max(abs(i), abs(j), abs(k)) == n
                ],
                dtype=np.float64,
            )
            np.testing.assert_array_equal(lattice_shell_steps(n), loops)


class RouteTest(unittest.TestCase):
    def test_moving_cells_with_lda_plus_u_or_ts_take_the_accelerated_path(self):
        from ..cli import _reference_only_features

        for name in ("ag", "h4"):
            for mode in ("Skip_force: .false.\n", "Minimization: bfgs\n", "Molecular_Dynamics: .true.\n"):
                with self.subTest(name=name, mode=mode):
                    features = _reference_only_features(
                        _translation(name, mode), accelerated_backend_selected=True
                    )
                    self.assertEqual(features, [])
        # spin takes the reference solver by its own entry (k-point workers, device fields there)
        spin = _translation("ag", "Spin_Polarization: .true.\nMinimization: bfgs\n")
        features = _reference_only_features(spin, accelerated_backend_selected=True)
        self.assertTrue(features)
        self.assertFalse(any("LDA+U" in item for item in features), features)

    def test_lda_plus_u_of_a_spin_polarized_wire_or_slab_takes_the_kpoint_driver(self):
        from types import SimpleNamespace

        from parsec_python import cli

        translation = SimpleNamespace(
            problem=SimpleNamespace(
                pseudopotentials={"Ag": SimpleNamespace(has_hubbard=True)},
                grid=SimpleNamespace(periodicity=2),
            )
        )
        with mock.patch.object(cli, "parallel_backend_uses_device", return_value=True):
            self.assertTrue(cli._gamma_spin_on_kpoint_driver(translation, True, False))
            with mock.patch.dict(os.environ, {"PARSEC_WIRE_SLAB_GAMMA_SPIN": "gamma"}):
                self.assertFalse(cli._gamma_spin_on_kpoint_driver(translation, True, False))


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class BandTest(unittest.TestCase):
    """LDA+U bands on the CuPy k-point workers against the CPU workers (CHEBDAV to 1e-10)."""

    def test_bands_with_the_term_equal_the_cpu_workers(self):
        from parsec_python.SCF.bands import band_mode, band_potentials, solve_kpoint_bands
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.SCF.single_point import run_scf
        from parsec_python.SCF.spin_polarized import run_scf_spin_polarized

        for extra in ("", "Spin_Polarization: .true.\nInitial_Spin_Polarization: 0.3\n"):
            with self.subTest(spin=bool(extra)):
                problem = _translation("ag", extra).problem
                system = prepare_periodic_single_point(problem)
                result = (run_scf_spin_polarized if extra else run_scf)(system)
                mode = band_mode(problem)
                potentials = band_potentials(result, mode)
                self.assertIn("hubbard", potentials)
                system = dataclasses.replace(
                    system,
                    input=dataclasses.replace(
                        system.input,
                        eigensolver=dataclasses.replace(
                            system.input.eigensolver, method="chebdav", tolerance=1.0e-10
                        ),
                    ),
                )
                reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(problem.periodic_cell.lattice_vectors)).T
                k_points = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.5, 0.0]]) @ reciprocal
                bands = {}
                for backend in ("cpu", "cupy"):
                    with mock.patch.dict(os.environ, {"PARSEC_KPOINT_WORKERS": "1"}):
                        bands[backend] = solve_kpoint_bands(
                            system,
                            k_points,
                            mode=mode,
                            potentials=potentials,
                            number_of_states=16,
                            backend=backend,
                            residual_tolerance=1.0e-7,
                        ).eigenvalues[..., :14]
                np.testing.assert_allclose(bands["cupy"], bands["cpu"], rtol=0.0, atol=1.0e-10)
                plain = {key: value for key, value in potentials.items() if key != "hubbard"}
                without = solve_kpoint_bands(
                    system, k_points, mode=mode, potentials=plain, number_of_states=16, backend="cupy"
                ).eigenvalues[..., :14]
                self.assertGreater(np.max(np.abs(without - bands["cpu"])), 1.0e-2)


if __name__ == "__main__":
    unittest.main()
