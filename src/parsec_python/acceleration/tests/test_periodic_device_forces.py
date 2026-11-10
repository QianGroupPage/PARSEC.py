"""Forces, relaxation and dynamics of periodic cells with their ionic fields on the device.

``acceleration/V_ion/periodic_device.py`` sums the local potential and the atomic densities of a
periodic cell over the images within the reach of their terms, on the device;
``acceleration/Forces/periodic_device.py`` differentiates them there and assembles the forces of the
accelerated Gamma-point path.  Checked here: the fields equal the readable ones to round-off, every
local term is the derivative of the device field it belongs to (central differences) and equals the
readable term, the projector gradients found through the grid-ball query equal those of the dense
scan bit for bit, the forces of the accelerated path are the derivative of its free energy and
equal the reference forces, and the accelerated command line runs single points with forces,
relaxations and dynamics of a bulk cell at Gamma itself (with the reference solver's trajectory)
and keeps the k-point inputs on the reference solver with the device fields.
"""


# With both sets of routines: the periodic inputs and grids of this module are those of the Fortran-derived
# routines (Use_Fortran_Routines: .true., PeriodicGridSettings(fortran_routines=True)); the numbers checked
# here are those of the Fortran parity round.
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Forces import periodic as periodic_forces
from parsec_python.Forces.periodic import (
    build_periodic_projector_gradients,
    periodic_local_forces,
)
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.models import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.V_ion import ewald as ewald_module
from parsec_python.V_ion import ionic_potential as ionic_module
from parsec_python.V_ion import load_pseudopotentials, superpose_atomic_density
from parsec_python.V_ion.ionic_potential import dense_support, set_support_finder

from ..Forces import periodic_device as device_forces
from ..V_ion import periodic_device as device_fields
from ..V_ion import periodic_setup_fortran as periodic_setup
from ..backends.cupy import cupy_available

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
STEP = 1.0e-6

_CUBE = PeriodicCell(lattice_vectors=5.6 * np.eye(3))
_FCC = PeriodicCell(
    lattice_vectors=0.5 * 10.26 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
)
_ATOMS = (Atom("Si", np.array([0.31, 0.22, 0.13])), Atom("Si", np.array([2.9, 2.6, 3.1])))


def _moved(atoms, index, axis, step):
    out = list(atoms)
    position = np.array(out[index].position, dtype=float)
    position[axis] += step
    out[index] = Atom(out[index].symbol, position)
    return tuple(out)


def _central(function, atoms, index, axis, step=STEP):
    return -(
        function(_moved(atoms, index, axis, step)) - function(_moved(atoms, index, axis, -step))
    ) / (2.0 * step)


def _fields(grid, lattice):
    x = grid.coordinates
    b = 2.0 * np.pi * np.linalg.inv(lattice).T
    density = 0.05 + 0.02 * np.cos(x @ b[0]) + 0.015 * np.sin(x @ (b[1] + b[2]))
    residual = 0.03 * np.sin(x @ b[1]) + 0.02 * np.cos(x @ (b[0] + b[1]))
    xc = -0.4 + 0.1 * np.cos(x @ b[2])
    return density, residual, xc


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceFieldTests(unittest.TestCase):
    """The device fields against the readable builders."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.si = {"Si": SpeciesPotential(DATA / "Si_POTRE.DAT", 2)}
        cls.si_potentials = load_pseudopotentials(cls.si)
        cls.core = {"H": SpeciesPotential(DATA / "H_CORE_POTRE.DAT", 0)}
        cls.core_potentials = load_pseudopotentials(cls.core)
        cls.grid = build_periodic_grid(_CUBE, PeriodicGridSettings(fortran_routines=True, spacing=0.7, expansion_order=4))

    def test_local_potential_equals_the_fft_split(self) -> None:
        lattice = _CUBE.lattice_vectors
        device = device_fields.device_local_ionic_potential(
            self.grid, _ATOMS, self.si_potentials, self.si, lattice, None, 1.0e-14
        )
        readable = periodic_setup.fft_local_ionic_potential(
            self.grid, _ATOMS, self.si_potentials, self.si, lattice, None, 1.0e-14
        )
        np.testing.assert_allclose(device, readable, rtol=0.0, atol=1.0e-12)
        # The terms it leaves out start where the table of r V is the Coulomb tail.
        reach = device_fields.coulomb_radius(self.si_potentials["Si"], 2)
        self.assertLess(reach, 10.0)

    def test_densities_equal_the_readable_sums(self) -> None:
        lattice = _CUBE.lattice_vectors
        atoms = tuple(Atom("H", atom.position) for atom in _ATOMS)
        for potentials, specifications, chosen, core in (
            (self.si_potentials, self.si, _ATOMS, False),
            (self.core_potentials, self.core, atoms, False),
            (self.core_potentials, self.core, atoms, True),
        ):
            with self.subTest(core=core, species=list(specifications)):
                device = device_fields.device_atomic_density(
                    self.grid, chosen, potentials, specifications, lattice, core=core
                )
                readable = superpose_atomic_density(
                    self.grid, chosen, potentials, specifications, lattice, core=core
                )
                np.testing.assert_allclose(device, readable, rtol=0.0, atol=1.0e-14)

    def test_a_shifted_grid_gives_the_readable_fields(self) -> None:
        # PARSEC 1.5's parity round: the orthorhombic grid of the Fortran-derived routines reads its shift
        lattice = _CUBE.lattice_vectors
        for shift in ((0.5, 0.5, 0.5), (0.25, 0.0, 0.5)):
            grid = build_periodic_grid(
                _CUBE, PeriodicGridSettings(fortran_routines=True, spacing=0.7, expansion_order=4, shift=shift)
            )
            with self.subTest(shift=shift):
                device = device_fields.device_local_ionic_potential(
                    grid, _ATOMS, self.si_potentials, self.si, lattice, None, 1.0e-14
                )
                readable = periodic_setup.fft_local_ionic_potential(
                    grid, _ATOMS, self.si_potentials, self.si, lattice, None, 1.0e-14
                )
                np.testing.assert_allclose(device, readable, rtol=0.0, atol=1.0e-12)
                # the points relative to the atoms of the grid without shift and the atoms moved back
                moved = tuple(Atom(atom.symbol, atom.position - np.asarray(shift) * 0.7) for atom in _ATOMS)
                back = device_fields.device_local_ionic_potential(
                    self.grid, moved, self.si_potentials, self.si, lattice, None, 1.0e-14
                )
                np.testing.assert_allclose(device, back, rtol=0.0, atol=1.0e-11)
                density = device_fields.device_atomic_density(grid, _ATOMS, self.si_potentials, self.si, lattice)
                np.testing.assert_allclose(
                    density,
                    superpose_atomic_density(grid, _ATOMS, self.si_potentials, self.si, lattice),
                    rtol=0.0,
                    atol=1.0e-14,
                )

    def test_what_the_device_builders_decline(self) -> None:
        general = build_periodic_grid(_FCC, PeriodicGridSettings(fortran_routines=True, spacing=0.9, expansion_order=4))
        self.assertIsNone(
            device_fields.device_local_ionic_potential(
                general, _ATOMS, self.si_potentials, self.si, _FCC.lattice_vectors, None, 1.0e-14
            )
        )
        # The atomic density of an oblique cell (the general grid of the Fortran-derived routines) is no longer
        # declined: the lattice kernel of device_lattice_atomic_density sums it (round 2026-11-08), equal to the
        # host sum to round-off (one thread per point adds the images in another order).
        density = device_fields.device_atomic_density(
            general, _ATOMS, self.si_potentials, self.si, _FCC.lattice_vectors
        )
        self.assertIsNotNone(density)
        with mock.patch.dict(os.environ, {"PARSEC_DEVICE": "cpu"}):
            self.assertIsNone(
                device_fields.device_atomic_density(general, _ATOMS, self.si_potentials, self.si, _FCC.lattice_vectors)
            )
            host = superpose_atomic_density(general, _ATOMS, self.si_potentials, self.si, _FCC.lattice_vectors)
        scale = float(np.max(np.abs(host)))
        self.assertGreater(scale, 0.0)
        np.testing.assert_allclose(density, host, rtol=0.0, atol=1.0e-14 * scale)
        splined = {"Si": SpeciesPotential(DATA / "Si_POTRE.DAT", 2, use_spline=True)}
        self.assertIsNone(
            device_fields.device_local_ionic_potential(
                self.grid, _ATOMS, self.si_potentials, splined, _CUBE.lattice_vectors, None, 1e-14
            )
        )
        self.assertIsNone(
            device_fields.device_atomic_density(self.grid, _ATOMS, self.si_potentials, self.si, None)
        )

    def test_scope_installs_the_device_builders_and_restores_the_others(self) -> None:
        before = (
            ewald_module._local_potential_provider,
            ewald_module._provider_converged_for,
            ionic_module._atomic_density_provider,
            periodic_forces._local_force_provider,
        )
        with device_forces.device_periodic_forces() as active:
            self.assertTrue(active)
            self.assertTrue(device_fields.device_preparation_active())
            self.assertIsNotNone(ionic_module._atomic_density_provider)
            self.assertIsNotNone(periodic_forces._local_force_provider)
        after = (
            ewald_module._local_potential_provider,
            ewald_module._provider_converged_for,
            ionic_module._atomic_density_provider,
            periodic_forces._local_force_provider,
        )
        self.assertEqual(before, after)
        # outside the scope: the host builder of the fast periodic preparation (the readable sum, same bits) and
        # the host forces of its Cubic_Spline potential (the readable sums for every other cell)
        from parsec_python.acceleration.Forces.periodic_spline import spline_periodic_local_forces

        self.assertIs(ionic_module._atomic_density_provider, periodic_setup._density_provider)
        self.assertIs(periodic_forces._local_force_provider, spline_periodic_local_forces)
        with mock.patch.dict(os.environ, {"PARSEC_DEVICE": "cpu"}):
            with device_forces.device_periodic_forces() as active:
                self.assertFalse(active)
                self.assertFalse(device_fields.device_preparation_active())

    def test_preparation_in_the_scope_is_the_readable_one_to_round_off(self) -> None:
        problem = SinglePointInput(
            atoms=list(_ATOMS),
            pseudopotentials=self.si,
            grid=PeriodicGridSettings(fortran_routines=True, spacing=0.7, expansion_order=4),
            periodic_cell=_CUBE,
            scf=SCFSettings(number_of_states=6),
        )
        readable = prepare_periodic_single_point(problem)
        with device_forces.device_periodic_forces():
            device = prepare_periodic_single_point(problem)
        np.testing.assert_allclose(device.ionic_potential, readable.ionic_potential, atol=1e-12, rtol=0)
        np.testing.assert_allclose(device.initial_density, readable.initial_density, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(
            device.nonlocal_operator.projectors.toarray(), readable.nonlocal_operator.projectors.toarray()
        )
        self.assertEqual(device.ion_ion_energy, readable.ion_ion_energy)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceLocalForceTests(unittest.TestCase):
    """Each local term: the readable term, and the derivative of its device field."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.si = {"Si": SpeciesPotential(DATA / "Si_POTRE.DAT", 2)}
        cls.si_potentials = load_pseudopotentials(cls.si)
        cls.core = {"H": SpeciesPotential(DATA / "H_CORE_POTRE.DAT", 0)}
        cls.core_potentials = load_pseudopotentials(cls.core)
        cls.grid = build_periodic_grid(_CUBE, PeriodicGridSettings(fortran_routines=True, spacing=0.7, expansion_order=4))
        cls.lattice = _CUBE.lattice_vectors

    def test_terms_equal_the_readable_terms(self) -> None:
        density, residual, xc = _fields(self.grid, self.lattice)
        hydrogen = tuple(Atom("H", atom.position) for atom in _ATOMS)
        for potentials, specifications, atoms in (
            (self.si_potentials, self.si, _ATOMS),
            (self.core_potentials, self.core, hydrogen),
        ):
            with self.subTest(species=list(specifications)):
                device = device_forces.device_periodic_local_forces(
                    self.grid, atoms, potentials, specifications, self.lattice, density, residual, xc
                )
                readable = periodic_local_forces(
                    self.grid, atoms, potentials, specifications, self.lattice, density, residual, xc
                )
                for name in ("ionic", "self_consistency", "core_correction"):
                    np.testing.assert_allclose(
                        getattr(device, name), getattr(readable, name), rtol=0.0, atol=1e-11, err_msg=name
                    )
                self.assertTrue(np.any(device.self_consistency != 0.0))

    def test_a_shifted_grid_gives_the_readable_terms(self) -> None:
        # the long-range term takes the phase of the shifted points (PARSEC 1.5's parity round)
        hydrogen = tuple(Atom("H", atom.position) for atom in _ATOMS)
        for shift in ((0.5, 0.5, 0.5), (0.25, 0.0, 0.5)):
            grid = build_periodic_grid(
                _CUBE, PeriodicGridSettings(fortran_routines=True, spacing=0.7, expansion_order=4, shift=shift)
            )
            density, residual, xc = _fields(grid, self.lattice)
            for potentials, specifications, atoms in (
                (self.si_potentials, self.si, _ATOMS),
                (self.core_potentials, self.core, hydrogen),
            ):
                with self.subTest(shift=shift, species=list(specifications)):
                    device = device_forces.device_periodic_local_forces(
                        grid, atoms, potentials, specifications, self.lattice, density, residual, xc
                    )
                    readable = periodic_local_forces(
                        grid, atoms, potentials, specifications, self.lattice, density, residual, xc
                    )
                    for name in ("ionic", "self_consistency", "core_correction"):
                        np.testing.assert_allclose(
                            getattr(device, name), getattr(readable, name), rtol=0.0, atol=1e-11, err_msg=name
                        )

    def test_ionic_term_is_the_derivative_of_the_device_potential(self) -> None:
        density, _, _ = _fields(self.grid, self.lattice)
        zeros = np.zeros(self.grid.size)
        terms = device_forces.device_periodic_local_forces(
            self.grid, _ATOMS, self.si_potentials, self.si, self.lattice, density, zeros, zeros
        )

        def energy(moved):
            field = device_fields.device_local_ionic_potential(
                self.grid, moved, self.si_potentials, self.si, self.lattice, None, 1.0e-14
            )
            return self.grid.integrate(density * field)

        for index in range(2):
            for axis in range(3):
                self.assertAlmostEqual(
                    terms.ionic[index, axis], _central(energy, _ATOMS, index, axis), delta=3e-8
                )

    def test_density_terms_are_the_derivatives_of_the_device_densities(self) -> None:
        _, residual, xc = _fields(self.grid, self.lattice)
        atoms = tuple(Atom("H", atom.position) for atom in _ATOMS)
        terms = device_forces.device_periodic_local_forces(
            self.grid, atoms, self.core_potentials, self.core, self.lattice,
            np.zeros(self.grid.size), residual, xc,
        )
        for axis in range(3):
            core = _central(
                lambda moved: self.grid.integrate(
                    xc * device_fields.device_atomic_density(
                        self.grid, moved, self.core_potentials, self.core, self.lattice, core=True
                    )
                ),
                atoms, 0, axis,
            )
            valence = _central(
                lambda moved: self.grid.integrate(
                    residual * device_fields.device_atomic_density(
                        self.grid, moved, self.core_potentials, self.core, self.lattice
                    )
                ),
                atoms, 0, axis,
            )
            self.assertAlmostEqual(terms.core_correction[0, axis], core, delta=3e-8)
            self.assertAlmostEqual(terms.self_consistency[0, axis], valence, delta=3e-8)


class ProjectorGradientRowTests(unittest.TestCase):
    def test_grid_ball_rows_give_the_gradients_of_the_dense_scan(self) -> None:
        si = {"Si": SpeciesPotential(DATA / "Si_POTRE.DAT", 2)}
        potentials = load_pseudopotentials(si)
        grid = build_periodic_grid(_CUBE, PeriodicGridSettings(fortran_routines=True, spacing=0.7, expansion_order=4))
        lattice = _CUBE.lattice_vectors
        reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
        for k_point in (None, 0.3 * reciprocal[0] - 0.2 * reciprocal[2]):
            with self.subTest(k=k_point is not None):
                fast = build_periodic_projector_gradients(grid, _ATOMS, potentials, si, lattice, k_point)
                set_support_finder(dense_support)
                try:
                    dense = build_periodic_projector_gradients(
                        grid, _ATOMS, potentials, si, lattice, k_point
                    )
                finally:
                    set_support_finder(periodic_setup.grid_ball)
                for left, right in zip(fast.components, dense.components):
                    np.testing.assert_array_equal(left.toarray(), right.toarray())
                self.assertEqual(fast.labels, dense.labels)


def _hydrogen_cell(*, criterion=1.0e-10):
    return SinglePointInput(
        atoms=[Atom("H", np.array([1.6, 2.4, 2.2])), Atom("H", np.array([3.05, 2.55, 2.45]))],
        pseudopotentials={"H": SpeciesPotential(DATA / "H_CORE_POTRE.DAT", 0)},
        grid=PeriodicGridSettings(fortran_routines=True, spacing=0.5, expansion_order=4),
        periodic_cell=PeriodicCell(lattice_vectors=5.0 * np.eye(3)),
        scf=SCFSettings(
            max_iterations=80,
            number_of_states=4,
            convergence_criterion=criterion,
            fermi_temperature_kelvin=300.0,
        ),
        hartree=HartreeSettings(relative_tolerance=1.0e-12, absolute_tolerance=1.0e-14, max_iterations=4000),
        eigensolver=EigensolverSettings(method="chebff", tolerance=criterion),
    )


def _free_energy(result, temperature=300.0):
    """``E - T S`` (the free energy the forces differentiate), as in ``tests/unit/test_periodic_forces.py``."""

    occupations = np.clip(np.asarray(result.occupations, dtype=float), 1e-300, 1 - 1e-16)
    degeneracy = 2.0 if getattr(result, "spin", None) is None else 1.0
    weights = np.ones_like(occupations)
    if getattr(result, "k_weights", None) is not None:
        weights = np.repeat(result.k_weights, occupations.size // len(result.k_weights))
    entropy = occupations * np.log(occupations) + (1 - occupations) * np.log(1 - occupations)
    return float(
        result.energies.total + 6.33327186e-6 * temperature * degeneracy * np.sum(weights * entropy)
    )


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class AcceleratedGammaTests(unittest.TestCase):
    """The forces of the accelerated Gamma-point path of a bulk cell."""

    def backend(self):
        from ..driver import ionic_backend

        return device_forces.with_periodic_operations(ionic_backend())

    def prepare(self, problem):
        from ..driver import prepare_single_point

        return prepare_single_point(problem, backend="cupy", symmetry="off")

    def test_forces_equal_the_reference_forces(self) -> None:
        from parsec_python.Relaxation.backend import fortran_periodic_backend as periodic_backend

        problem = _hydrogen_cell()
        reference_backend = periodic_backend()
        reference_system = prepare_periodic_single_point(problem)
        reference = reference_backend.compute_forces(
            reference_system, reference_backend.run_scf(reference_system), symmetrize=False
        )
        with device_forces.device_periodic_forces():
            system = self.prepare(problem)
            backend = self.backend()
            result = backend.run_scf(system)
            forces = backend.compute_forces(system, result, symmetrize=False)
        self.assertTrue(forces.periodic)
        np.testing.assert_allclose(forces.unsymmetrized, reference.unsymmetrized, rtol=0, atol=1e-7)
        np.testing.assert_allclose(forces.local.core_correction, reference.local.core_correction, atol=1e-7)
        self.assertTrue(np.any(forces.local.core_correction != 0.0))
        self.assertTrue(np.any(forces.nonlocal_pseudopotential != 0.0))

    def test_forces_are_the_derivative_of_the_free_energy(self) -> None:
        problem = _hydrogen_cell()
        step = 1.0e-5
        with device_forces.device_periodic_forces():
            system = self.prepare(problem)
            backend = self.backend()
            forces = backend.compute_forces(system, backend.run_scf(system), symmetrize=False)
            for index, axis in ((1, 0), (1, 2)):
                energies = []
                for sign in (1.0, -1.0):
                    moved = backend.update_geometry(
                        system, _moved(problem.atoms, index, axis, sign * step)
                    )
                    energies.append(_free_energy(backend.run_scf(moved)))
                numeric = -(energies[0] - energies[1]) / (2.0 * step)
                self.assertAlmostEqual(forces.unsymmetrized[index, axis], numeric, delta=2e-6)

    def test_moved_atoms_are_a_fresh_preparation(self) -> None:
        problem = _hydrogen_cell()
        moved_atoms = (Atom("H", np.array([-0.4, 2.4, 2.2])), problem.atoms[1])
        with device_forces.device_periodic_forces():
            system = self.prepare(problem)
            moved = self.backend().update_geometry(system, moved_atoms)
            fresh = prepare_periodic_single_point(
                SinglePointInput(
                    atoms=list(moved_atoms),
                    pseudopotentials=problem.pseudopotentials,
                    grid=problem.grid,
                    periodic_cell=problem.periodic_cell,
                    scf=problem.scf,
                    hartree=problem.hartree,
                    eigensolver=problem.eigensolver,
                )
            )
        np.testing.assert_array_equal(moved.atoms[0].position, [4.6, 2.4, 2.2])
        for name in ("ionic_potential", "initial_density", "core_density"):
            np.testing.assert_array_equal(getattr(moved, name), getattr(fresh, name))
        np.testing.assert_array_equal(
            moved.nonlocal_operator.projectors.toarray(), fresh.nonlocal_operator.projectors.toarray()
        )


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class ReferenceSolverInTheScopeTests(unittest.TestCase):
    """The reference periodic backend (k-points, collinear spin) with the device fields and forces."""

    def check(self, problem, sampling=None, components=((1, 0), (1, 2)), step=1e-5, delta=2e-6):
        from parsec_python.Relaxation.backend import fortran_periodic_backend as periodic_backend
        from parsec_python.SCF.pbc import update_periodic_geometry

        backend = periodic_backend(sampling)
        with device_forces.device_periodic_forces():
            system = prepare_periodic_single_point(problem)
            result = backend.run_scf(system)
            forces = backend.compute_forces(system, result, symmetrize=False)
            for index, axis in components:
                energies = []
                for sign in (1.0, -1.0):
                    moved = update_periodic_geometry(system, _moved(problem.atoms, index, axis, sign * step))
                    energies.append(_free_energy(backend.run_scf(moved)))
                numeric = -(energies[0] - energies[1]) / (2.0 * step)
                self.assertAlmostEqual(forces.unsymmetrized[index, axis], numeric, delta=delta)
        outside = backend.compute_forces(
            prepare_periodic_single_point(problem),
            backend.run_scf(prepare_periodic_single_point(problem)),
            symmetrize=False,
        )
        np.testing.assert_allclose(forces.unsymmetrized, outside.unsymmetrized, rtol=0, atol=1e-7)

    def test_kpoints(self) -> None:
        reciprocal = 2.0 * np.pi / 5.0
        k_points = np.array([[0.0, 0.0, 0.25 * reciprocal], [0.0, 0.0, -0.25 * reciprocal]])
        self.check(_hydrogen_cell(), (k_points, np.array([0.5, 0.5])))

    def test_collinear_spin_at_gamma(self) -> None:
        from dataclasses import replace

        problem = _hydrogen_cell()
        problem = replace(
            problem,
            atoms=[*problem.atoms, Atom("H", np.array([2.3, 3.9, 1.2]))],
            scf=replace(problem.scf, spin_polarized=True),
        )
        self.check(problem, components=((2, 1),))


_CLI_INPUT = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  5.0 5.0 5.0
end Cell_Shape
Grid_Spacing: 0.5
Expansion_Order: 4
Ignore_Symmetry: .true.
States_Num: 4
Fermi_Temp: 300.0
Max_Iter: 80
Convergence_Criterion: 1.0e-9
Diag_Tolerance: 1.0e-9
Mixing_Method: Anderson
Mixing_Param: 0.3
Correlation_Type: ca
Atom_Types_Num: 1
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
  1.6  2.4  2.2
  3.05 2.55 2.45
end Atom_Coord
"""

_TWO_K = (
    "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\nbegin Kpoint_List\n"
    "  0.0 0.0 0.25 1.0\n  0.0 0.0 -0.25 1.0\nend Kpoint_List\n"
)


class MovingCellOptionRouteTests(unittest.TestCase):
    def test_forces_with_the_cell_options_stay_on_the_reference_solver(self) -> None:
        from parsec_python.Input.parsec_input import parse_parsec_input
        from parsec_python.acceleration.cli import _reference_only_features

        def features(extra: str) -> list[str]:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
                directory = Path(name)
                shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
                path = directory / "parsec.in"
                text = _CLI_INPUT
                if "Van_Der_Waals" in extra:
                    # PARSEC's TS correction is that of GGA-PBE (error 158).
                    text = text.replace("Correlation_Type: ca", "Correlation_Type: pb")
                path.write_text(text + extra, encoding="utf-8")
                translation = parse_parsec_input(path)
            return _reference_only_features(translation, accelerated_backend_selected=True)

        self.assertEqual(features("Minimization: bfgs\n"), [])
        self.assertEqual(features("Calc_Hartree_G: .true.\n"), [])
        # TS (and LDA+U, test_periodic_correction_forces.py) move on the accelerated path.
        self.assertEqual(features("Van_Der_Waals: .true.\nMinimization: bfgs\n"), [])
        # Fortran's G-space local potential moves on the accelerated path too (its ionic force is gspace_local_forces,
        # the other terms the ball sums of Forces/periodic_local.py::fortran_spline_local_forces)
        self.assertEqual(features("Periodic_Local_Potential: g_space\nMinimization: bfgs\n"), [])
        for option in (
            "Calc_Hartree_G: .true.\n",
        ):
            with self.subTest(option=option):
                routed = features(option + "Minimization: bfgs\n")
                self.assertTrue(
                    any("molecular dynamics of a periodic cell with" in item for item in routed),
                    routed,
                )
        # Zero_Average_Vxc, the default of a 3-D cell since 2026-11-08, moves on the accelerated path (its forces read
        # the shifted potentials of the result through the reference formula)
        self.assertEqual(features("Zero_Average_Vxc: .true.\nMinimization: bfgs\n"), [])


class ScopeTests(unittest.TestCase):
    def test_only_forces_and_moving_atoms_of_a_periodic_cell_take_the_device_fields(self) -> None:
        from contextlib import nullcontext
        from types import SimpleNamespace

        from parsec_python.Input.parsec_input import parse_parsec_input

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            directory = Path(name)
            shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
            path = directory / "parsec.in"
            path.write_text(_CLI_INPUT, encoding="utf-8")
            plain = parse_parsec_input(path)
            path.write_text(_CLI_INPUT + "Minimization: bfgs\n", encoding="utf-8")
            moving = parse_parsec_input(path)
        self.assertIsInstance(device_forces.periodic_ionic_scope(plain), nullcontext)
        self.assertNotIsInstance(device_forces.periodic_ionic_scope(moving), nullcontext)
        cluster = SimpleNamespace(
            problem=SimpleNamespace(periodic_cell=None),
            compute_forces=True,
            relaxation=None,
            molecular_dynamics=None,
        )
        self.assertIsInstance(device_forces.periodic_ionic_scope(cluster), nullcontext)

    def test_mpi_still_refuses_periodic_forces(self) -> None:
        from .. import cli as accelerated_cli
        from ..Parallel import mpi

        for extra in ("Skip_force: .false.\n", _TWO_K + "Minimization: bfgs\n"):
            with self.subTest(kpoints=extra.startswith("Kpoint")):
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
                    directory = Path(name)
                    shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
                    path = directory / "parsec.in"
                    path.write_text(_CLI_INPUT + extra, encoding="utf-8")
                    errors = StringIO()
                    with mock.patch.object(accelerated_cli, "_join_mpi", return_value=None), mock.patch.object(
                        accelerated_cli, "_abort_mpi"
                    ), mock.patch.object(
                        mpi, "requested", return_value=True
                    ), redirect_stderr(errors), redirect_stdout(StringIO()):
                        code = accelerated_cli.main([str(path), "--quiet", "--no-archive"])
                self.assertEqual(code, 2)
                self.assertIn("forces, relaxation or molecular dynamics of a periodic", errors.getvalue())
                self.assertIn("--mpi", errors.getvalue())


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class CommandLineTests(unittest.TestCase):
    """The accelerated command line against the reference one."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)

    def run_both(self, extra: str):
        from parsec_python.cli import main as reference_main

        from ..cli import main as accelerated_main

        archives, messages = [], []
        for name, main in (("accelerated", accelerated_main), ("reference", reference_main)):
            folder = self.directory / name
            folder.mkdir()
            shutil.copy(DATA / "H_POTRE.DAT", folder / "H_POTRE.DAT")
            path = folder / "parsec.in"
            path.write_text(_CLI_INPUT + extra, encoding="utf-8")
            errors = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(errors):
                code = main([str(path), "--quiet"])
            self.assertIn(code, (0, 4), errors.getvalue())
            messages.append(errors.getvalue())
            with np.load(folder / "parsec_python_results.npz", allow_pickle=True) as archive:
                archives.append({key: archive[key] for key in archive.files})
        return archives, messages

    def test_single_point_forces_run_on_the_accelerated_path(self) -> None:
        (accelerated, reference), (message, _) = self.run_both("Skip_force: .false.\n")
        self.assertNotIn("reference parsec_python.cli solver", message)
        self.assertEqual(str(accelerated["backend_selected"]), "cupy")
        np.testing.assert_allclose(accelerated["forces_total"], reference["forces_total"], atol=1e-7)
        self.assertAlmostEqual(
            float(accelerated["energy_total_ry"]), float(reference["energy_total_ry"]), delta=1e-8
        )

    def test_relaxation_follows_the_reference_trajectory(self) -> None:
        (accelerated, reference), _ = self.run_both(
            "Minimization: bfgs\nMovement_Num: 2\nForce_Min: 0.0005\n"
        )
        for key in ("relaxation_positions", "relaxation_energies", "relaxation_forces"):
            np.testing.assert_allclose(accelerated[key], reference[key], rtol=0, atol=1e-7, err_msg=key)

    def test_dynamics_follows_the_reference_trajectory(self) -> None:
        (accelerated, reference), _ = self.run_both(
            "Molecular_Dynamics: .true.\nStep_num: 2\nTinit: 300.0\nT_Final: 300.0\n"
            "T_Step: 0.0\nTime_Step: 20.0\ntest_rng: .true.\n"
        )
        keys = [key for key in reference if key.startswith("dynamics_") and reference[key].dtype.kind == "f"]
        self.assertTrue(keys)
        for key in keys:
            np.testing.assert_allclose(accelerated[key], reference[key], rtol=0, atol=1e-7, err_msg=key)

    def test_kpoint_forces_stay_with_the_reference_solver_and_its_device_fields(self) -> None:
        (accelerated, reference), (message, _) = self.run_both(_TWO_K + "Skip_force: .false.\n")
        self.assertIn("k-point sampling", message)
        np.testing.assert_allclose(accelerated["forces_total"], reference["forces_total"], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
