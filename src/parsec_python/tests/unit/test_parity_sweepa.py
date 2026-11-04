"""The final parity sweep (part A) of the 2026-10-27 round.

* Wire/slab ionic steps with a reduced k-point set: the bulk rules of
  movements a wedge cannot follow (molecular dynamics with a weighted list, a
  relaxation whose Move_Flag breaks the group) hold for wires and slabs.
* Wire/slab relaxation, dynamics and polarizability with collinear spin and
  k-points (the spin k-point loop starts from the channel potentials of the
  structure or field before).
* Wire/slab LDA+U with k-points, spin and bands (Bloch projectors replicated
  along the periodic directions alone, occupations averaged over the
  wire/slab group of ``alatt``).
* Spin-orbit forces on the double grid (gradients of the double-grid
  spin-orbit projectors).
* Warm starts of the spinor drivers between ionic steps (``vnew - vion_old +
  vion_new``, the saved spinors of the isolated spin-orbit loops).
* Spinor Polarizability with forces (the forces of the last field loop, as
  ``parsec.F90`` computes them after ``polar.f90``).

The comparisons with Fortran PARSEC are in the report of the parity round.
"""


# With both sets of routines: the periodic grids of this module are those of the Fortran-derived routines
# (PeriodicGridSettings(fortran_routines=True), as Use_Fortran_Routines: .true. gives); the numbers checked
# here are those of the Fortran parity round.
from __future__ import annotations

import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from parsec_python.Input import UnsupportedParsecOptionError, parse_parsec_input

DATA = Path(__file__).resolve().parents[1] / "data"
HYDROGEN = DATA / "water_H_POTRE.DAT"
AG = DATA / "Ag_POTRE.DAT"
AUH = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH"


class _Directory(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.setdefault("PARSEC_KPOINT_BACKEND", "cpu")
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)

    def write(self, text: str, name: str = "parsec.in") -> Path:
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def run_cli(self, path: Path) -> int:
        from parsec_python.cli import main

        return main([str(path), "--quiet"])

    def scf_loops(self) -> list[int]:
        """Iterations of every SCF loop of parsec.out (``m- i ... SRE`` lines)."""

        loops: list[int] = []
        for line in (self.directory / "parsec.out").read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*\d+-\s*(\d+)\s+(up |dn )?.*SRE", line)
            if match and match.group(2) in (None, "up "):
                iteration = int(match.group(1))
                if iteration == 1 or not loops:
                    loops.append(iteration)
                else:
                    loops[-1] = iteration
        return loops


# ---------------------------------------------------------------------------
# Item 1: wire/slab movements with a reduced k-point set
# ---------------------------------------------------------------------------

_H2_PERIODIC = """\
Boundary_Conditions: {boundary}
begin Cell_Shape
{cell}
end Cell_Shape
Boundary_Sphere_Radius: 4.0
Grid_Spacing: 0.6
States_Num: 2
Coordinate_Unit: Cartesian_Bohr
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
{flag}
begin Atom_Coord
  0.0 0.0 0.7
  0.0 0.0 -0.7
end Atom_Coord
"""

_WEIGHTED = (
    "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\nbegin Kpoint_List\n"
    " 0.0 0.0 0.0 1.0\n 0.25 0.0 0.0 2.0\n 0.5 0.0 0.0 1.0\nend Kpoint_List\n"
)
_EQUAL = _WEIGHTED.replace("2.0", "1.0")
_SHIFT = (
    "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n 4\nend Monkhorst_Pack_Grid\n"
    "begin Monkhorst_Pack_Shift\n 0.5\nend Monkhorst_Pack_Shift\n"
)
_MD = "Molecular_Dynamics: .true.\nStep_num: 2\n"


class WireSlabReducedKpointMovementTests(_Directory):
    """The bulk refusals (``_check_reduced_kpoint_movements``) for
    wires and slabs: the wedge forces are completed over the wire/slab group,
    which molecular dynamics drops and the ionic loop restricts to the
    operations Move_Flag respects."""

    def parse(self, extra: str, *, boundary: str = "wire", cell: str = " 3.0", flag: str = ""):
        shutil.copy(HYDROGEN, self.directory / "H_POTRE.DAT")
        return parse_parsec_input(
            self.write(_H2_PERIODIC.format(boundary=boundary, cell=cell, extra=extra, flag=flag))
        )

    def test_dynamics_with_a_weighted_list(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedParsecOptionError, "Molecular_Dynamics and manual movements use no symmetry"
        ):
            self.parse(_WEIGHTED + _MD)
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "manual movements use no symmetry"):
            self.write("  0.0 0.0 0.8\n  0.0 0.0 -0.8\n", "manual.dat")
            self.parse(_WEIGHTED + "Minimization: manual\nMovement_Num: 1\n")
        # Equal weights, Ignore_Symmetry, the time-reversal wedge of
        # Monkhorst_Pack_Shift and a slab (no group in PARSEC without bands,
        # spin-orbit coupling or NSCF) are accepted.
        self.assertIsNotNone(self.parse(_EQUAL + _MD).molecular_dynamics)
        self.assertIsNotNone(
            self.parse(_WEIGHTED + _MD + "Ignore_Symmetry: .true.\n").molecular_dynamics
        )
        self.assertIsNotNone(self.parse(_SHIFT + _MD).molecular_dynamics)
        slab = self.parse(_WEIGHTED + _MD, boundary="slab", cell=" 3.0 0.0\n 0.0 3.2")
        self.assertIsNotNone(slab.molecular_dynamics)

    def test_relaxation_whose_move_flag_breaks_the_group(self) -> None:
        # The mirror z -> -z of the wire exchanges the two atoms; Move_Flag 1
        # moves the first only.
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Move_Flag fixes atoms"):
            self.parse(_SHIFT + "Minimization: bfgs\n", flag="Move_Flag: 1")
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Move_Flag fixes atoms"):
            self.parse(_WEIGHTED + "Minimization: bfgs\n", flag="Move_Flag: 1")
        # Every atom movable, Ignore_Symmetry, or an unreduced set: accepted.
        self.assertEqual(self.parse(_SHIFT + "Minimization: bfgs\n").relaxation.method, "bfgs")
        self.assertEqual(
            self.parse(
                _SHIFT + "Minimization: bfgs\nIgnore_Symmetry: .true.\n", flag="Move_Flag: 1"
            ).relaxation.method,
            "bfgs",
        )
        self.assertEqual(
            self.parse(_EQUAL + "Minimization: bfgs\n", flag="Move_Flag: 1").relaxation.method,
            "bfgs",
        )


# ---------------------------------------------------------------------------
# Item 2: wire/slab relaxation, dynamics and polarizability with spin + k
# ---------------------------------------------------------------------------

_H3_SPIN = """\
Boundary_Conditions: {boundary}
begin Cell_Shape
{cell}
end Cell_Shape
Boundary_Sphere_Radius: 4.5
Grid_Spacing: 0.6
Coordinate_Unit: Cartesian_Bohr
States_Num: 5
Convergence_Criterion: 1.0e-5
Max_Iter: 80
Spin_Polarization: .true.
Correlation_Type: ca
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
   0.00   0.10   0.65
   1.40   0.00  -0.75
  -1.90   0.30   0.20
end Atom_Coord
"""


class WireSlabSpinKpointTests(_Directory):
    def setUp(self) -> None:
        super().setUp()
        shutil.copy(HYDROGEN, self.directory / "H_POTRE.DAT")

    def test_ionic_warm_start_of_the_spin_kpoint_loop(self) -> None:
        """``vnew(:,isp) - vion_old + vion_new`` of both channels (parsec.F90)."""

        from parsec_python.Relaxation.backend import wire_slab_backend

        backend = wire_slab_backend((np.zeros((1, 3)), np.ones(1)))
        rng = np.random.default_rng(2)
        old, new = rng.normal(size=6), rng.normal(size=6)
        up, down = rng.normal(size=6), rng.normal(size=6)
        start = backend.warm_start(
            SimpleNamespace(ionic_potential=old),
            SimpleNamespace(next_potentials=(up, down)),
            SimpleNamespace(ionic_potential=new),
        )
        np.testing.assert_allclose(start[0], up - old + new, atol=1e-15)
        np.testing.assert_allclose(start[1], down - old + new, atol=1e-15)

    def test_polarizability_field_start_of_the_spin_kpoint_loop(self) -> None:
        """``pot%vnew = pot%vold`` of both channels between two field loops."""

        from parsec_python.SCF.polarizability import warm_start_for_next_field

        rng = np.random.default_rng(3)
        inputs = (rng.normal(size=5), rng.normal(size=5))
        start = warm_start_for_next_field(
            SimpleNamespace(
                k_points=np.zeros((2, 3)),
                density_up=np.zeros(5),
                scf_input_potentials=inputs,
                scf_hartree_potential=np.ones(5),
            )
        )
        self.assertEqual(len(start.input_potential), 2)
        np.testing.assert_array_equal(start.input_potential[1], inputs[1])
        self.assertIsNone(start.eigensolver_state)

    def test_slab_movements_with_spin_and_kpoints(self) -> None:
        # Manual movements (one atom moved by 0.03 bohr): the second structure
        # starts from the channel potentials of the first.
        self.write(
            "   0.00   0.10   0.68\n   1.40   0.00  -0.75\n  -1.90   0.30   0.20\n", "manual.dat"
        )
        path = self.write(
            _H3_SPIN.format(
                boundary="slab",
                cell=" 5.0 0.0\n 0.0 5.0",
                extra=_SHIFT.replace(" 4\n", " 2 2\n").replace(" 0.5\n", " 0.5 0.5\n")
                + "Minimization: manual\nMovement_Num: 1\n",
            )
        )
        self.assertEqual(self.run_cli(path), 0)
        text = (self.directory / "atom.cor").read_text(encoding="utf-8")
        self.assertEqual(text.count("Step #"), 2)
        loops = self.scf_loops()
        self.assertEqual(len(loops), 2)
        self.assertLess(loops[1], loops[0] - 2)
        with np.load(self.directory / "parsec_python_results.npz") as archive:
            self.assertIn("relaxation_energies", archive.files)
            self.assertEqual(archive["forces_total"].shape, (3, 3))

    def test_wire_polarizability_with_spin_kpoints_and_forces(self) -> None:
        path = self.write(
            _H3_SPIN.format(
                boundary="wire",
                cell=" 6.0",
                extra=(
                    "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\n"
                    "begin Kpoint_List\n 0.125 0.0 0.0 1.0\n 0.375 0.0 0.0 1.0\nend Kpoint_List\n"
                    "Polarizability: .true.\nElectric_Field: 0.005\nSkip_Force: .false.\n"
                ),
            )
        )
        self.assertEqual(self.run_cli(path), 0)
        lines = (self.directory / "polar.dat").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len([line for line in lines if line[:12].count(".") == 3]), 7)
        loops = self.scf_loops()
        self.assertEqual(len(loops), 7)
        # Every field loop after the first starts from the one before.
        self.assertLess(max(loops[1:]), loops[0])
        with np.load(self.directory / "parsec_python_results.npz") as archive:
            self.assertEqual(archive["forces_total"].shape, (3, 3))


# ---------------------------------------------------------------------------
# Item 3: wire/slab LDA+U with k-points, spin and bands
# ---------------------------------------------------------------------------


def _box_grid(sides, spacing):
    axes = [np.arange(0.0, side - 1e-9, spacing) for side in sides]
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return SimpleNamespace(
        coordinates=coordinates,
        volume_element=spacing**3,
        size=coordinates.shape[0],
        spacing=spacing,
        settings=SimpleNamespace(double_grid_order=1),
        shape=tuple(len(axis) for axis in axes),
    )


_AG_WIRE = """\
Boundary_Conditions: wire
begin Cell_Shape
  6.0
end Cell_Shape
Boundary_Sphere_Radius: 5.0
Grid_Spacing: 0.6
Coordinate_Unit: Cartesian_Bohr
States_Num: 10
Fermi_Temp: 1000.0
Convergence_Criterion: 1.0e-7
Max_Iter: 80
Eigensolver: chebdav
Correlation_Type: ca
{extra}
Atom_Types_Num: 1
Atom_Type: Ag
Local_Component: s
ldaplusu_u 4.0 eV
begin Atom_Coord
  0.0 0.2 0.1
end Atom_Coord
"""


class WireSlabHubbardKpointTests(_Directory):
    def setUp(self) -> None:
        super().setUp()
        shutil.copy(AG, self.directory / "Ag_POTRE.DAT")

    def system(self, extra: str = ""):
        from parsec_python.SCF import prepare_periodic_single_point

        translation = parse_parsec_input(self.write(_AG_WIRE.format(extra=extra)))
        return translation, prepare_periodic_single_point(translation.problem)

    def test_replica_and_symmetry_lattices(self) -> None:
        from parsec_python.Hamiltonian.hubbard import (
            hubbard_replica_lattice,
            hubbard_symmetry_lattice,
            periodic_replica_translations,
        )

        _, system = self.system()
        rows = hubbard_replica_lattice(system)
        np.testing.assert_array_equal(rows, [[6.0, 0.0, 0.0]])
        self.assertEqual(periodic_replica_translations(rows).shape, (3, 3))
        np.testing.assert_array_equal(np.diag(hubbard_symmetry_lattice(system)), [6.0, 20.0, 20.0])
        # The Gamma projectors of the wire are those of the replica rows.
        from parsec_python.Hamiltonian.hubbard import build_hubbard_projectors

        rebuilt = build_hubbard_projectors(
            system.grid, system.atoms, system.pseudopotentials, system.input.pseudopotentials, rows
        )
        np.testing.assert_array_equal(rebuilt.projectors.toarray(), system.hubbard.projectors.toarray())

    def test_bloch_projectors_of_a_wire_fold_onto_the_doubled_wire(self) -> None:
        """k = pi/a on the wire equals Gamma of the wire of period 2a on Bloch functions."""

        from parsec_python.Hamiltonian.hubbard import (
            HubbardOperator,
            build_hubbard_projectors,
            kpoint_occupation_matrix,
            occupation_matrices,
        )
        from parsec_python.Pseudopotential import read_parsec_pseudopotential
        from parsec_python.models import Atom, SpeciesPotential

        potential = {"Ag": read_parsec_pseudopotential(AG)}
        specifications = {"Ag": SpeciesPotential(AG, 0, element_symbol="Ag", hubbard_u=0.3)}
        side, spacing = 4.2, 0.6
        position = np.array([0.7, 3.1, 4.0])
        primitive = _box_grid((side, 8.4, 8.4), spacing)
        doubled = _box_grid((2 * side, 8.4, 8.4), spacing)
        k_point = np.array([np.pi / side, 0.0, 0.0])
        # One periodic row: no replica along the confined y and z.
        primitive_projectors = build_hubbard_projectors(
            primitive, (Atom("Ag", position),), potential, specifications,
            np.array([[side, 0.0, 0.0]]), k_point=k_point,
        )
        doubled_projectors = build_hubbard_projectors(
            doubled, (Atom("Ag", position), Atom("Ag", position + [side, 0, 0])),
            potential, specifications, np.array([[2 * side, 0.0, 0.0]]),
        )
        rng = np.random.default_rng(7)
        u = rng.normal(size=(primitive.size, 3)) + 1j * rng.normal(size=(primitive.size, 3))
        n = primitive.shape[0]
        index = np.arange(doubled.size)
        ix, rest = np.divmod(index, primitive.shape[1] * primitive.shape[2])
        rows = (ix % n) * primitive.shape[1] * primitive.shape[2] + rest
        phase = np.exp(1j * doubled.coordinates @ k_point)
        psi = phase[:, None] * u[rows]
        np.testing.assert_allclose(
            doubled_projectors.overlaps(psi)[0], primitive_projectors.overlaps(u)[0], atol=1e-12
        )
        occupations = np.array([1.0, 0.4, 0.1])
        n_k = kpoint_occupation_matrix(primitive_projectors, u, occupations, 1.0)
        n_doubled = occupation_matrices(doubled_projectors, psi, occupations, keep_complex=True)
        np.testing.assert_allclose(n_doubled[1], n_k[0], atol=1e-11)
        matrix = rng.normal(size=(5, 5))
        matrix = matrix + matrix.T
        applied_k = HubbardOperator(primitive_projectors, matrix[None]).apply(u)
        applied = HubbardOperator(doubled_projectors, np.stack([matrix, matrix])).apply(psi)
        np.testing.assert_allclose(applied, phase[:, None] * applied_k[rows], atol=1e-11)
        # The three rows of the wire's cell (unit vectors along the confined
        # directions) are no lattice for the replicas: upot.F90's images one
        # cell away would sit one bohr off the atom.
        with self.assertRaisesRegex(ValueError, "exceeds a cell side"):
            build_hubbard_projectors(
                primitive, (Atom("Ag", position),), potential, specifications,
                np.diag([side, 1.0, 1.0]), k_point=k_point,
            )

    def test_kpoint_gamma_equals_the_gamma_loops(self) -> None:
        """The Bloch LDA+U of a wire at k = 0 is its Gamma LDA+U (scalar and spin)."""

        from parsec_python.SCF import run_scf
        from parsec_python.SCF.kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized
        from parsec_python.SCF.spin_polarized import run_scf_spin_polarized

        k0 = (
            "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\n"
            "begin Kpoint_List\n 0.0 0.0 0.0 1.0\nend Kpoint_List\n"
        )
        for spin in (False, True):
            with self.subTest(spin=spin):
                extra = "Spin_Polarization: .true.\nIgnore_Symmetry: .true.\n" if spin else "Ignore_Symmetry: .true.\n"
                _, gamma_system = self.system(extra)
                gamma = (run_scf_spin_polarized if spin else run_scf)(gamma_system)
                _, k_system = self.system(extra + k0)
                driver = run_scf_kpoints_spin_polarized if spin else run_scf_kpoints
                kpoint = driver(k_system, np.zeros((1, 3)), np.ones(1))
                self.assertTrue(gamma.converged and kpoint.converged)
                self.assertAlmostEqual(kpoint.energies.total, gamma.energies.total, delta=2e-7)
                self.assertAlmostEqual(kpoint.hubbard.energy, gamma.hubbard.energy, delta=2e-6)
                self.assertGreater(kpoint.hubbard.energy, 0.0)

    def test_wedge_occupations_use_the_wire_group(self) -> None:
        from parsec_python.SCF.kpoint_hubbard import KPointHubbard
        from parsec_python.Hamiltonian.hubbard import symmetrize_occupations

        translation, system = self.system(_SHIFT.replace(" 4\n", " 2\n"))
        group = translation.problem.space_group
        self.assertGreater(group.order, 1)
        state = KPointHubbard.for_system(system, np.zeros((1, 3)))
        self.assertEqual(len(state.operations), group.order)
        rng = np.random.default_rng(5)
        block = rng.normal(size=(1, 1, 5, 5))
        block = block + np.swapaxes(block, -1, -2)
        once = symmetrize_occupations(block, state.operations)
        np.testing.assert_allclose(symmetrize_occupations(once, state.operations), once, atol=1e-12)

    def test_bands_of_a_wire_with_lda_plus_u(self) -> None:
        """The band Hamiltonian (Bloch projectors of the band k-point and the
        SCF's D) gives the SCF levels at the SCF k-points."""

        from parsec_python.SCF.bands import band_potentials, solve_kpoint_bands
        from parsec_python.SCF.kpoints import run_scf_kpoints

        k_list = (
            "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\n"
            "begin Kpoint_List\n 0.0 0.0 0.0 1.0\n 0.25 0.0 0.0 1.0\nend Kpoint_List\n"
            "Ignore_Symmetry: .true.\n"
        )
        _, system = self.system(k_list)
        k_points = np.array([[0.0, 0.0, 0.0], [0.25 * 2.0 * np.pi / 6.0, 0.0, 0.0]])
        result = run_scf_kpoints(system, k_points, np.full(2, 0.5))
        self.assertTrue(result.converged)
        bands = solve_kpoint_bands(
            system,
            k_points,
            mode="scalar",
            potentials=band_potentials(result, "scalar"),
            number_of_states=6,
        )
        scf_levels = np.asarray(result.eigenvalues).reshape(2, -1)[:, :6]
        np.testing.assert_allclose(bands.eigenvalues[0], scf_levels, atol=2e-5)


# ---------------------------------------------------------------------------
# Item 4: spin-orbit forces on the double grid
# ---------------------------------------------------------------------------


class SpinOrbitDoubleGridForceTests(unittest.TestCase):
    def test_double_grid_spin_orbit_term_is_the_derivative_of_its_energy(self) -> None:
        from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
        from parsec_python.Forces.spin_orbit_fortran import (
            build_spin_orbit_projector_gradients,
            spin_orbit_energy,
            spin_orbit_forces,
        )
        from parsec_python.Grid.pbc import build_periodic_grid
        from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
        from parsec_python.V_ion import load_pseudopotentials

        species = {
            "Au": SpeciesPotential(AUH / "Au_POTRE.DAT", 0, spin_orbit=True),
            "H": SpeciesPotential(AUH / "H_POTRE.DAT", 0),
        }
        atoms = (Atom("Au", np.array([3.4, 3.1, 3.3])), Atom("H", np.array([0.6, 3.5, 2.9])))
        cell = PeriodicCell(lattice_vectors=6.5 * np.eye(3))
        grid = build_periodic_grid(
            cell, PeriodicGridSettings(fortran_routines=True, spacing=0.5, expansion_order=4, double_grid_order=2)
        )
        self.assertEqual(grid.settings.double_grid_order, 2)
        potentials = load_pseudopotentials(species)
        rng = np.random.default_rng(1)
        spinors = rng.normal(size=(2 * grid.size, 4)) + 1j * rng.normal(size=(2 * grid.size, 4))
        spinors /= np.linalg.norm(spinors, axis=0)
        occupations = np.array([1.0, 0.7, 0.3, 0.05])
        step = 1e-6

        def moved(axis, delta):
            position = np.array(atoms[0].position, dtype=float)
            position[axis] += delta
            return (Atom("Au", position), atoms[1])

        for lattice, k_point in (
            (None, None),
            (cell.lattice_vectors, None),
            (cell.lattice_vectors, np.array([0.2, -0.1, 0.3])),
        ):
            with self.subTest(periodic=lattice is not None, k_point=k_point is not None):
                projectors = build_spin_orbit_projectors(
                    grid, atoms, potentials, species, lattice, k_point
                )
                gradients = build_spin_orbit_projector_gradients(
                    grid, atoms, potentials, species, lattice, k_point
                )
                for projector, gradient in zip(projectors, gradients):
                    np.testing.assert_array_equal(projector.support_rows, gradient.support_rows)
                forces = spin_orbit_forces(projectors, gradients, spinors, occupations, 2)
                np.testing.assert_array_equal(forces[1], 0.0)
                for axis in range(3):
                    energies = [
                        spin_orbit_energy(
                            build_spin_orbit_projectors(
                                grid, moved(axis, sign * step), potentials, species, lattice, k_point
                            ),
                            spinors,
                            occupations,
                        )
                        for sign in (1.0, -1.0)
                    ]
                    numeric = -(energies[0] - energies[1]) / (2 * step)
                    self.assertAlmostEqual(forces[0, axis], numeric, delta=1e-10)


# ---------------------------------------------------------------------------
# Items 5 and 6: spinor warm starts and spinor Polarizability with forces
# ---------------------------------------------------------------------------

_AUH = """\
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.6
Expansion_Order: 6
States_Num: 20
Fermi_Temp: 500.0
Max_Iter: 60
Convergence_Criterion: 1.0e-5
FF_MaxIter: 9
Chebdav_Degree: 30
Mixing_Param: 0.25
Eigensolver: chebff
SO_from_scratch: .true.
{extra}
Atom_Types_Num: 2
Atom_Type: Au
SO_PSP: .true.
Local_Component: s
Move_Flag: 0
Begin atom_coord
   1.445 0 0
End atom_coord
Atom_Type: H
Local_Component: s
Move_Flag: 0
Begin atom_coord
   -1.445 0 0
End atom_coord
Correlation_Type: CA
"""


class SpinorWarmStartTests(_Directory):
    def setUp(self) -> None:
        super().setUp()
        for name in ("Au_POTRE.DAT", "H_POTRE.DAT"):
            shutil.copy(AUH / name, self.directory / name)

    # (test_warm_start_replaces_the_ionic_part tested the parity round's spinor_warm_start, which the merged tree
    # replaced by Andrew Grieder's warm start: tests/unit/test_spinor_forces.py SpinOrbitWarmStartTests)

    def test_spin_orbit_loop_continues_from_its_own_state(self) -> None:
        """The last mixed potential and the saved spinors of a converged loop
        start a loop of the same structure at convergence."""

        from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
        from parsec_python.SCF import prepare_single_point
        from parsec_python.SCF.self_consistent_soc import run_self_consistent_soc

        translation = parse_parsec_input(self.write(_AUH.format(extra="")))
        system = prepare_single_point(translation.problem)
        projectors = tuple(
            build_spin_orbit_projectors(
                system.grid, system.atoms, system.pseudopotentials, system.input.pseudopotentials
            )
        )
        first = run_self_consistent_soc(system, projectors)
        self.assertTrue(first.converged)
        self.assertIsNotNone(first.next_potential)
        self.assertIsNotNone(first.eigensolver_state)
        again = run_self_consistent_soc(
            system,
            projectors,
            initial_potential=first.next_potential,
            initial_eigensolver_state=first.eigensolver_state,
        )
        self.assertLessEqual(again.iterations, 2)
        self.assertAlmostEqual(again.energies.total, first.energies.total, delta=1e-6)

    def test_noncollinear_loop_takes_a_starting_potential_and_field(self) -> None:
        from parsec_python.SCF import prepare_single_point
        from parsec_python.SCF.noncollinear import run_self_consistent_noncollinear

        shutil.copy(HYDROGEN, self.directory / "H_POTRE.DAT")
        text = (
            "Boundary_Sphere_Radius: 5.0\nGrid_Spacing: 0.5\nStates_Num: 4\n"
            "Fermi_Temp: 500.0\nMax_Iter: 60\nConvergence_Criterion: 1.0e-6\n"
            "Eigensolver: chebdav\nNon_Collinear_magnetism: .true.\nCorrelation_Type: ca\n"
            "Atom_Types_Num: 1\nAtom_Type: H\nLocal_Component: s\n"
            "begin Atom_Coord\n 0.0 0.0 -0.7\n 0.0 0.0 0.7\nend Atom_Coord\n"
        )
        system = prepare_single_point(parse_parsec_input(self.write(text)).problem)
        cold, warm = [], []
        first = run_self_consistent_noncollinear(system, callback=cold.append)
        self.assertTrue(first.converged)
        potential, field = first.next_potential
        self.assertEqual(field.shape, (3, system.grid.size))
        again = run_self_consistent_noncollinear(
            system, initial_potential=(potential, field), callback=warm.append
        )
        self.assertTrue(again.converged)
        # The first eigenproblem (random vectors: the loop keeps no subspace)
        # already sees the converged potential and field.
        self.assertLess(warm[0].weighted_residual, 0.1 * cold[0].weighted_residual)
        self.assertLess(again.iterations, first.iterations)
        self.assertAlmostEqual(again.energies.total, first.energies.total, delta=1e-6)

    def test_spin_orbit_relaxation_starts_from_the_structure_before(self) -> None:
        self.write("   1.460 0.0 0.0\n  -1.445 0.0 0.0\n", "manual.dat")
        path = self.write(
            _AUH.format(extra="Minimization: manual\nMovement_Num: 1").replace("Move_Flag: 0\n", "")
        )
        self.assertEqual(self.run_cli(path), 0)
        loops = self.scf_loops()
        self.assertEqual(len(loops), 2)
        self.assertLess(loops[1], loops[0] - 3)

    def test_spinor_polarizability_with_forces(self) -> None:
        """parsec.F90 computes the forces after the seven loops of polar.f90:
        those of the last one (field along -z), spin-orbit term included."""

        path = self.write(
            _AUH.format(
                extra="Polarizability: .true.\nElectric_Field: 0.005\nSkip_Force: .false."
            ).replace("1.0e-5", "5.0e-3")
        )
        self.assertIn(self.run_cli(path), (0, 3))
        text = (self.directory / "polar.dat").read_text(encoding="utf-8")
        self.assertIn("Polarizability of the cluster", text)
        log = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertEqual(log.count("Converged spin-orbit-split spinor states"), 7)
        self.assertEqual(log.count(" Adding spin-orbit contribution to forces"), 1)
        # Every field loop after the first starts from the loop before (its
        # last mixed potential and spinors), as polar.f90 keeps vold and the
        # wavefunctions.
        loops = self.scf_loops()
        self.assertEqual(len(loops), 7)
        self.assertLess(max(loops[1:]), loops[0])
        with np.load(self.directory / "parsec_python_results.npz") as archive:
            self.assertEqual(archive["forces_spin_orbit"].shape, (2, 3))
            self.assertEqual(archive["forces_total"].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
