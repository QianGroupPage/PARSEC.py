"""A ``Monkhorst_Pack_Grid`` without a ``Monkhorst_Pack_Shift`` block, as PARSEC 1.5 reduces it (2026-11-18).

PARSEC 1.5 (``usrinputfile.F90:976-994``) reads ``mpshift`` only from the block; without it the shift keeps its
static zero, and ``init_var.F90:281-289`` calls ``irrbz`` for every Monkhorst-Pack input: the Gamma-centred grid
``(i-1)/n`` reduced by the space group and time reversal, with the density symmetrized (``usrinputfile.F90:2548``).
Checked here:

* the port's ``irrbz`` against a literal transliteration of ``irrbz.f90`` on unshifted meshes (cubic, oblique fcc,
  orthorhombic of low symmetry; with and without time reversal);
* the tables PARSEC 1.5 printed for such inputs (the WSL binary of the Fortran parity round, 2026-11-18: Si8 cube,
  a Si8 cube with a moved atom, a stretched cell, meshes 4 4 4, 3 3 3, 2 2 1, 2 2 2 and 3 4 2 without a shift block,
  and 4 4 4 with a zero block, which prints the same table);
* the parser: on either set of routines an input without the block gives the k-points, weights, group and density
  symmetrization of the same input with a zero ``Monkhorst_Pack_Shift`` block (without choosing the Fortran-derived
  routines as the block does); wires and slabs; a hybrid; a mesh of one point;
* ``PARSEC_MP_UNSHIFTED_MESH=port``: the former unreduced mesh ``(2i-n-1)/(2n)``, bit for bit.
"""

from __future__ import annotations

import itertools
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Grid.kpoints import (
    fold_to_centered_zone,
    irreducible_monkhorst_pack,
    kpoint_sampling,
    kpoint_sampling_details,
    monkhorst_pack_grid,
    time_reversal_reduced,
    unshifted_mesh_setting,
)
from parsec_python.Grid.space_group import find_space_group, fortran_reciprocal_lattice
from parsec_python.Input.parsec_input import parse_parsec_input
from parsec_python.models import Atom

_ROOT = Path(__file__).resolve().parents[4]
_SI_POTRE = _ROOT / "examples" / "3d_Si" / "Si_POTRE.DAT"
_H_POTRE = _ROOT / "src" / "parsec_python" / "tests" / "data" / "H_POTRE.DAT"
_SI8 = [
    (0.0, 0.0, 0.0), (0.0, 5.13, 5.13), (5.13, 0.0, 5.13), (5.13, 5.13, 0.0),
    (2.565, 2.565, 2.565), (2.565, 7.695, 7.695), (7.695, 2.565, 7.695), (7.695, 7.695, 2.565),
]
_FCC = 10.26 / 2.0 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
_PORT = {"PARSEC_MP_UNSHIFTED_MESH": "port"}


def _fortran_irrbz(lattice, mpgrid, mpshift, gmtrx, ncl=False):
    """``irrbz.f90`` line by line (PARSEC 1.5): ``(kibz, kpwt, kptrep, kptgrid)``, ``kibz`` unfolded in ``[0, 1)``."""

    b_rows = fortran_reciprocal_lattice(lattice)
    bvec = [[float(b_rows[j][i]) for j in range(3)] for i in range(3)]  # bvec(i, j): component i of b_j
    tol_bz = 1.0e-8

    def matvec3(mat, vec):
        return [sum(float(mat[i][j]) * vec[j] for j in range(3)) for i in range(3)]

    def dot(vec):
        return sum(value * value for value in vec)

    def anint(value):
        return math.copysign(math.floor(abs(value) + 0.5), value)

    def translate01(kvec):
        out = []
        for value in kvec:
            k_tmp = math.fmod(value + 0.5, 1.0) - 0.5  # Fortran mod: the sign of its first argument
            if abs(k_tmp) < tol_bz:
                k_tmp = 0.0
            if k_tmp < 0.0:
                k_tmp = k_tmp + 1.0
            out.append(k_tmp)
        return out

    kibz, kptrep, ekin = [], [], []
    kptgrid = np.zeros(mpgrid, dtype=np.int64)
    ikpt = 0
    for ii in range(mpgrid[0]):
        for jj in range(mpgrid[1]):
            for ll in range(mpgrid[2]):
                kprot = [(float(index) + shift) / float(count) for index, shift, count in zip((ii, jj, ll), mpshift, mpgrid)]
                new_kpt = True
                for matrix in gmtrx:
                    krot = translate01(matvec3(matrix, kprot))
                    if ikpt == 0:
                        ikpt = 1
                        kibz.append(list(krot))
                        ekin.append(dot(matvec3(bvec, [k - anint(k) for k in krot])))
                        kptrep.append(0)
                        kptgrid[ii, jj, ll] = 0
                    for itest in range(ikpt):
                        for a, inv in enumerate((1.0, -1.0), start=1):
                            if ncl and a != 1:
                                continue
                            ktest = translate01([k - inv * q for k, q in zip(krot, kibz[itest])])
                            norm2 = dot(matvec3(bvec, ktest))
                            if norm2 < tol_bz and new_kpt:
                                new_kpt = False
                                norm2 = dot(matvec3(bvec, [k - anint(k) for k in kprot])) + 8.0
                                kptgrid[ii, jj, ll] = itest
                                norm1 = dot(matvec3(bvec, kibz[itest]))
                                if norm2 < norm1:
                                    kibz[itest] = translate01(kprot)
                                    ekin[itest] = norm2
                                kptrep[itest] += 1
                if new_kpt:
                    ikpt += 1
                    kibz.append(list(kprot))
                    ekin.append(dot(matvec3(bvec, [k - anint(k) for k in krot])))
                    kptrep.append(1)
                    kptgrid[ii, jj, ll] = ikpt - 1
    total = int(np.prod(mpgrid))
    assert sum(kptrep) == total
    return np.array(kibz), np.array(kptrep, dtype=float) / total, np.array(kptrep), kptgrid


# PARSEC 1.5's "K-point generation" tables (irrbz.f90:180-200; weight and relative coordinates) of the Si8 inputs of
# item4_scripts/make_fortran_cases.py, from the WSL binary of the Fortran parity round (gfortran 13.3, 2026-11-18).
# Rows: weight (format f11.7) and the relative coordinates of qpoints.dat (3f14.10).
_FORTRAN_TABLES: dict[str, list[tuple[float, float, float, float]]] = {
    "si_k444": [
        (0.0156250, 0.0, 0.0, 0.0),
        (0.0937500, 0.0, 0.0, 0.25),
        (0.0468750, 0.0, 0.0, 0.5),
        (0.1875000, 0.0, 0.25, 0.25),
        (0.1875000, 0.0, 0.25, 0.5),
        (0.0468750, 0.0, 0.5, 0.5),
        (0.1250000, 0.25, 0.25, 0.25),
        (0.1875000, 0.25, 0.25, 0.5),
        (0.0937500, 0.25, 0.5, 0.5),
        (0.0156250, 0.5, 0.5, 0.5),
    ],
    "si_k444_s0": [
        (0.0156250, 0.0, 0.0, 0.0),
        (0.0937500, 0.0, 0.0, 0.25),
        (0.0468750, 0.0, 0.0, 0.5),
        (0.1875000, 0.0, 0.25, 0.25),
        (0.1875000, 0.0, 0.25, 0.5),
        (0.0468750, 0.0, 0.5, 0.5),
        (0.1250000, 0.25, 0.25, 0.25),
        (0.1875000, 0.25, 0.25, 0.5),
        (0.0937500, 0.25, 0.5, 0.5),
        (0.0156250, 0.5, 0.5, 0.5),
    ],
    "si_k333": [
        (0.0370370, 0.0, 0.0, 0.0),
        (0.2222222, 0.0, 0.0, 0.3333333333),
        (0.4444444, 0.0, 0.3333333333, 0.3333333333),
        (0.2962963, 0.3333333333, 0.3333333333, 0.3333333333),
    ],
    "si_k221": [
        (0.2500000, 0.0, 0.0, 0.0),
        (0.5000000, 0.0, 0.5, 0.0),
        (0.2500000, 0.5, 0.5, 0.0),
    ],
    "sim_k444": [
        (0.0156250, 0.0, 0.0, 0.0),
        (0.0312500, 0.0, 0.0, 0.25),
        (0.0156250, 0.0, 0.0, 0.5),
        (0.0312500, 0.0, 0.25, 0.0),
        (0.0312500, 0.0, 0.25, 0.25),
        (0.0312500, 0.0, 0.25, 0.5),
        (0.0312500, 0.0, 0.25, 0.75),
        (0.0156250, 0.0, 0.5, 0.0),
        (0.0312500, 0.0, 0.5, 0.25),
        (0.0156250, 0.0, 0.5, 0.5),
        (0.0312500, 0.25, 0.0, 0.0),
        (0.0312500, 0.25, 0.0, 0.25),
        (0.0312500, 0.25, 0.0, 0.5),
        (0.0312500, 0.25, 0.0, 0.75),
        (0.0312500, 0.25, 0.25, 0.0),
        (0.0312500, 0.25, 0.25, 0.25),
        (0.0312500, 0.25, 0.25, 0.5),
        (0.0312500, 0.25, 0.25, 0.75),
        (0.0312500, 0.25, 0.5, 0.0),
        (0.0312500, 0.25, 0.5, 0.25),
        (0.0312500, 0.25, 0.5, 0.5),
        (0.0312500, 0.25, 0.5, 0.75),
        (0.0312500, 0.25, 0.75, 0.0),
        (0.0312500, 0.25, 0.75, 0.25),
        (0.0312500, 0.25, 0.75, 0.5),
        (0.0312500, 0.25, 0.75, 0.75),
        (0.0156250, 0.5, 0.0, 0.0),
        (0.0312500, 0.5, 0.0, 0.25),
        (0.0156250, 0.5, 0.0, 0.5),
        (0.0312500, 0.5, 0.25, 0.0),
        (0.0312500, 0.5, 0.25, 0.25),
        (0.0312500, 0.5, 0.25, 0.5),
        (0.0312500, 0.5, 0.25, 0.75),
        (0.0156250, 0.5, 0.5, 0.0),
        (0.0312500, 0.5, 0.5, 0.25),
        (0.0156250, 0.5, 0.5, 0.5),
    ],
    "sim_k222": [
        (0.1250000, 0.0, 0.0, 0.0),
        (0.1250000, 0.0, 0.0, 0.5),
        (0.1250000, 0.0, 0.5, 0.0),
        (0.1250000, 0.0, 0.5, 0.5),
        (0.1250000, 0.5, 0.0, 0.0),
        (0.1250000, 0.5, 0.0, 0.5),
        (0.1250000, 0.5, 0.5, 0.0),
        (0.1250000, 0.5, 0.5, 0.5),
    ],
    "sio_k342": [
        (0.0416667, 0.0, 0.0, 0.0),
        (0.0416667, 0.0, 0.0, 0.5),
        (0.0833333, 0.0, 0.25, 0.0),
        (0.0833333, 0.0, 0.25, 0.5),
        (0.0416667, 0.0, 0.5, 0.0),
        (0.0416667, 0.0, 0.5, 0.5),
        (0.0833333, 0.3333333333, 0.0, 0.0),
        (0.0833333, 0.3333333333, 0.0, 0.5),
        (0.1666667, 0.3333333333, 0.25, 0.0),
        (0.1666667, 0.3333333333, 0.25, 0.5),
        (0.0833333, 0.3333333333, 0.5, 0.0),
        (0.0833333, 0.3333333333, 0.5, 0.5),
    ],
}


def _si8_atoms(moved=False, stretch=1.0):
    atoms = [list(position) for position in _SI8]
    if moved:
        atoms[0] = [0.10, 0.05, 0.0]
    return [Atom("Si", [x, y, z * stretch]) for x, y, z in atoms]


def _diamond_atoms():
    return [Atom("Si", list(position)) for position in _SI8]


class TransliterationTests(unittest.TestCase):
    """The port's irrbz (fold=False) is PARSEC 1.5's on unshifted meshes."""

    def check(self, lattice, atoms, meshes, ncl=False) -> None:
        group = find_space_group(lattice, atoms)
        for mesh in meshes:
            with self.subTest(mesh=mesh, order=group.order, ncl=ncl):
                kibz, weights, counts, grid = _fortran_irrbz(lattice, mesh, (0.0, 0.0, 0.0), group.reciprocal_rotations, ncl)
                ours = irreducible_monkhorst_pack(
                    lattice, mesh, (0.0, 0.0, 0.0), group.reciprocal_rotations, time_reversal=not ncl, fold=False
                )
                np.testing.assert_array_equal(ours.counts, counts)
                np.testing.assert_allclose(ours.fractional, kibz, rtol=0.0, atol=1.0e-14)
                np.testing.assert_allclose(ours.weights, weights, rtol=1.0e-14, atol=0.0)
                np.testing.assert_array_equal(ours.grid_map, grid)
                folded = irreducible_monkhorst_pack(
                    lattice, mesh, (0.0, 0.0, 0.0), group.reciprocal_rotations, time_reversal=not ncl
                )
                np.testing.assert_array_equal(folded.fractional, fold_to_centered_zone(ours.fractional))

    def test_cubic_diamond(self) -> None:
        lattice = np.diag([10.26] * 3)
        self.check(lattice, _diamond_atoms(), [(2, 2, 2), (3, 3, 3), (4, 4, 4), (2, 2, 1), (3, 4, 2)])
        self.check(lattice, _diamond_atoms(), [(4, 4, 4), (2, 2, 1)], ncl=True)

    def test_oblique_fcc(self) -> None:
        atoms = [Atom("Si", [0.0, 0.0, 0.0]), Atom("Si", [2.565, 2.565, 2.565])]
        self.check(_FCC, atoms, [(2, 2, 2), (3, 3, 3), (4, 4, 4), (2, 2, 1)])

    def test_low_symmetry(self) -> None:
        lattice = np.diag([10.26, 10.26, 11.25])
        self.check(lattice, _si8_atoms(moved=True, stretch=11.25 / 10.26), [(2, 2, 2), (3, 3, 3), (4, 4, 4), (2, 2, 1), (3, 4, 2)])


_SI_INPUT = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
   10.26 10.26 {c}
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.45
Expansion_Order: 12
States_Num: 24
Fermi_Temp: 300.0
Correlation_Type: {xc}
{extra}
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
{mesh}
end Monkhorst_Pack_Grid
{shift}Atom_Types_Num: 1
Atom_Type: Si
Local_Component: p
begin Atom_Coord
{atoms}end Atom_Coord
"""
_ZERO_SHIFT = "begin Monkhorst_Pack_Shift\n0.0 0.0 0.0\nend Monkhorst_Pack_Shift\n"


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        shutil.copy(_SI_POTRE, self.directory / "Si_POTRE.DAT")

    def parse(self, mesh="4 4 4", shift="", extra="", moved=False, stretch=1.0, xc="ca", environment=None):
        atoms = _si8_atoms(moved=moved, stretch=stretch)
        text = _SI_INPUT.format(
            c=10.26 * stretch,
            xc=xc,
            extra=extra,
            mesh=mesh,
            shift=shift,
            atoms="".join(f"  {a.position[0]:.6f} {a.position[1]:.6f} {a.position[2]:.6f}\n" for a in atoms),
        )
        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        with mock.patch.dict(os.environ, environment or {}):
            return parse_parsec_input(path)

    def assert_same_sampling(self, ours, theirs) -> None:
        k_ours, w_ours = kpoint_sampling(ours.problem)
        k_theirs, w_theirs = kpoint_sampling(theirs.problem)
        np.testing.assert_array_equal(k_ours, k_theirs)
        np.testing.assert_array_equal(w_ours, w_theirs)
        self.assertEqual(ours.problem.monkhorst_pack_shift, theirs.problem.monkhorst_pack_shift)
        self.assertEqual(ours.problem.monkhorst_pack_dimensions, theirs.problem.monkhorst_pack_dimensions)
        self.assertEqual(ours.problem.scf.symmetrize_density, theirs.problem.scf.symmetrize_density)
        groups = [problem.space_group for problem in (ours.problem, theirs.problem)]
        self.assertEqual([None if g is None else g.order for g in groups], [None if g is None else g.order for g in groups[::-1]])
        self.assertEqual(ours.problem.grid, theirs.problem.grid)

    def test_without_the_block_both_routines_take_the_zero_shift(self) -> None:
        cases = [
            ("4 4 4", False, 1.0),
            ("3 3 3", False, 1.0),
            ("2 2 1", False, 1.0),
            ("2 2 2", True, 1.0),
            ("4 4 4", True, 1.0),
            ("3 4 2", False, 11.25 / 10.26),
        ]
        for routines in ("Use_Fortran_Routines: .true.", "Use_Fortran_Routines: .false."):
            for mesh, moved, stretch in cases:
                with self.subTest(routines=routines, mesh=mesh, moved=moved, stretch=stretch):
                    unshifted = self.parse(mesh, "", routines, moved, stretch)
                    zero = self.parse(mesh, _ZERO_SHIFT, routines, moved, stretch)
                    self.assertEqual(unshifted.problem.monkhorst_pack_shift, (0.0, 0.0, 0.0))
                    group = unshifted.problem.space_group
                    # Fortran's routines symmetrize every k-point run; the default ones where an operation besides
                    # the identity maps their grid onto itself (else time reversal alone, as for the zero block)
                    self.assertEqual(
                        unshifted.problem.scf.symmetrize_density,
                        unshifted.routines.fortran or (group is not None and group.order > 1),
                    )
                    self.assert_same_sampling(unshifted, zero)
                    details = kpoint_sampling_details(unshifted.problem)
                    group = unshifted.problem.space_group
                    reference = _fortran_irrbz(
                        unshifted.problem.periodic_cell.lattice_vectors,
                        details.dimensions,
                        (0.0, 0.0, 0.0),
                        [np.eye(3, dtype=np.int64)] if group is None else group.reciprocal_rotations,
                    )
                    np.testing.assert_allclose(details.fractional, fold_to_centered_zone(reference[0]), atol=1e-14)
                    np.testing.assert_allclose(details.weights, reference[1], rtol=1e-14)

    def test_the_routines_are_chosen_as_before(self) -> None:
        # without Use_Fortran_Routines a mesh without the block keeps the default routines (the block alone chooses
        # the Fortran-derived ones by default); its wedge is that of the operations that map their grid onto itself
        unshifted = self.parse("4 4 4")
        self.assertFalse(unshifted.routines.fortran)
        self.assertEqual(len(kpoint_sampling(unshifted.problem)[0]), 10)
        self.assertTrue(any("PARSEC 1.5's Gamma-centred mesh" in note for note in unshifted.warnings))
        shifted = self.parse("4 4 4", _ZERO_SHIFT)
        self.assertTrue(shifted.routines.fortran)
        self.assertEqual(len(kpoint_sampling(shifted.problem)[0]), 10)
        with mock.patch.dict(os.environ, _PORT):
            self.assertFalse(self.parse("4 4 4").routines.fortran)

    def test_port_keeps_the_former_mesh_bit_for_bit(self) -> None:
        for routines in ("", "Use_Fortran_Routines: .true.", "Use_Fortran_Routines: .false."):
            for mesh in ("4 4 4", "3 3 3", "2 2 1"):
                with self.subTest(routines=routines, mesh=mesh):
                    former = self.parse(mesh, "", routines, environment=_PORT)
                    problem = former.problem
                    self.assertIsNone(problem.monkhorst_pack_shift)
                    self.assertIsNone(problem.kpoints)
                    self.assertIsNone(problem.space_group)
                    self.assertFalse(problem.scf.symmetrize_density)
                    dimensions = tuple(int(value) for value in mesh.split())
                    self.assertEqual(problem.monkhorst_pack_dimensions, dimensions)
                    expected = monkhorst_pack_grid(problem.periodic_cell, dimensions)
                    k_points, weights = kpoint_sampling(problem)
                    np.testing.assert_array_equal(k_points, expected[0])
                    np.testing.assert_array_equal(weights, expected[1])
                    self.assertFalse(any("PARSEC 1.5's Gamma-centred mesh" in note for note in former.warnings))
        with mock.patch.dict(os.environ, {"PARSEC_MP_UNSHIFTED_MESH": "nonsense"}):
            with self.assertRaises(ValueError):
                unshifted_mesh_setting()
        with mock.patch.dict(os.environ, {"PARSEC_MP_UNSHIFTED_MESH": ""}):
            self.assertEqual(unshifted_mesh_setting(), "fortran")

    def test_a_mesh_of_one_point_stays_the_gamma_point(self) -> None:
        for environment in ({}, _PORT):
            with self.subTest(environment=environment):
                translation = self.parse("1 1 1", environment=environment)
                problem = translation.problem
                self.assertIsNone(problem.monkhorst_pack_shift)
                self.assertEqual(problem.monkhorst_pack_dimensions, (1, 1, 1))
                self.assertFalse(problem.scf.symmetrize_density)
                k_points, weights = kpoint_sampling(problem)
                np.testing.assert_array_equal(k_points, np.zeros((1, 3)))
                np.testing.assert_array_equal(weights, [1.0])

    def test_hybrid_takes_time_reversal_of_the_zero_shift_mesh(self) -> None:
        for routines in ("Use_Fortran_Routines: .true.", "Use_Fortran_Routines: .false."):
            with self.subTest(routines=routines):
                unshifted = self.parse("3 3 3", "", routines, xc="pbe0")
                zero = self.parse("3 3 3", _ZERO_SHIFT, routines, xc="pbe0")
                k_ours, w_ours = kpoint_sampling(unshifted.problem)
                k_zero, w_zero = kpoint_sampling(zero.problem)
                np.testing.assert_array_equal(k_ours, k_zero)
                np.testing.assert_array_equal(w_ours, w_zero)
                self.assertEqual(len(k_ours), 14)  # (27 + 1)/2: Gamma and 13 pairs
                lattice = unshifted.problem.periodic_cell.lattice_vectors
                fractions = k_ours @ np.linalg.inv(2.0 * np.pi * np.linalg.inv(lattice).T)
                mesh = np.rint(fractions * 3.0)
                np.testing.assert_allclose(fractions * 3.0, mesh, atol=1e-10)

    def test_forces_take_the_wedge_or_the_pairs_of_the_zero_shift_mesh(self) -> None:
        moving = self.parse("2 2 2", "", "Skip_Force: .false.", moved=True)
        self.assertFalse(moving.routines.fortran)
        zero = self.parse("2 2 2", _ZERO_SHIFT, "Use_Fortran_Routines: .false.\nSkip_Force: .false.", moved=True)
        self.assert_same_sampling(moving, zero)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_FORCE_WEDGE": "off"}):
            pairs = self.parse("2 2 2", "", "Skip_Force: .false.", moved=True)
        self.assertTrue(pairs.problem.kpoint_time_reversal)
        lattice = pairs.problem.periodic_cell.lattice_vectors
        full = irreducible_monkhorst_pack(lattice, (2, 2, 2), (0.0, 0.0, 0.0), time_reversal=False)
        np.testing.assert_allclose(pairs.problem.kpoints, full.k_points, atol=1e-14)
        expected = time_reversal_reduced(full.k_points, full.weights, lattice)
        np.testing.assert_allclose(kpoint_sampling(pairs.problem)[0], expected[0], atol=1e-14)


class FallbackTests(unittest.TestCase):
    """Where the zero shift's wedge would be refused by name, time reversal alone (a zero block stops there)."""

    def test_incommensurate_translations(self) -> None:
        from parsec_python.Input.parsec_input import UnsupportedParsecOptionError
        from parsec_python.tests.unit.test_berkeleygw import PERIODIC, _parse

        mesh = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n  2 2 2\nend Monkhorst_Pack_Grid\n"
        translation = _parse(PERIODIC, mesh)
        problem = translation.problem
        self.assertEqual(problem.monkhorst_pack_shift, (0.0, 0.0, 0.0))
        self.assertIsNone(problem.space_group)
        self.assertFalse(problem.scf.symmetrize_density)
        self.assertTrue(any("reduced by time reversal alone" in line for line in translation.warnings))
        k_points, weights = kpoint_sampling(problem)
        self.assertEqual(len(k_points), 8)  # every point of the Gamma-centred 2^3 mesh is its own -k
        np.testing.assert_allclose(weights, np.full(8, 0.125))
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Symmetrize_Charge_Density"):
            _parse(PERIODIC, mesh + _ZERO_SHIFT)
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Symmetrize_Charge_Density"):
            _parse(PERIODIC, mesh + "Symmetrize_Charge_Density: .true.\n")
        with mock.patch.dict(os.environ, _PORT):
            former = _parse(PERIODIC, mesh).problem
        self.assertIsNone(former.monkhorst_pack_shift)
        self.assertIsNone(former.space_group)

    def test_dos_mesh_of_a_restart_without_dos_mp_shift(self) -> None:
        # Create_DOS with Restart_Run on a Gamma input: the restarted SCF runs on DOS_MP_Grid with PARSEC 1.5's zero
        # shift; where that wedge is refused (a C2 whose translation no grid spacing meets), time reversal alone,
        # as the SCF's mesh without Monkhorst_Pack_Shift (a DOS_MP_Shift block, or a written
        # Symmetrize_Charge_Density, still stops there by name)
        from parsec_python.Input.parsec_input import UnsupportedParsecOptionError
        from parsec_python.tests.unit.test_berkeleygw import PERIODIC, _parse

        dos = "Create_DOS: .true.\nRestart_Run: .true.\nbegin DOS_MP_Grid\n  2 2 2\nend DOS_MP_Grid\n"
        translation = _parse(PERIODIC, dos)
        problem = translation.problem
        self.assertEqual(problem.monkhorst_pack_dimensions, (2, 2, 2))
        self.assertEqual(problem.monkhorst_pack_shift, (0.0, 0.0, 0.0))
        self.assertIsNone(problem.space_group)
        self.assertFalse(problem.scf.symmetrize_density)
        self.assertTrue(
            any(line.startswith("DOS_MP_Grid without DOS_MP_Shift") for line in translation.warnings)
        )
        k_points, weights = kpoint_sampling(problem)
        self.assertEqual(len(k_points), 8)  # PARSEC 1.5's Gamma-centred 2^3 mesh, each point its own -k
        np.testing.assert_allclose(weights, np.full(8, 0.125))
        zero = "begin DOS_MP_Shift\n  0.0 0.0 0.0\nend DOS_MP_Shift\n"
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Symmetrize_Charge_Density"):
            _parse(PERIODIC, dos + zero)
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Symmetrize_Charge_Density"):
            _parse(PERIODIC, dos + "Symmetrize_Charge_Density: .true.\n")
        # PARSEC_MP_UNSHIFTED_MESH=port: the former unreduced grid of the DOS divisions
        with mock.patch.dict(os.environ, _PORT):
            former = _parse(PERIODIC, dos).problem
        self.assertIsNone(former.monkhorst_pack_shift)
        self.assertIsNone(former.space_group)
        self.assertEqual(former.monkhorst_pack_dimensions, (2, 2, 2))


class FortranTableTests(unittest.TestCase):
    """The k-points and weights PARSEC 1.5 printed for the same inputs."""

    def test_no_block_is_a_zero_block_in_parsec(self) -> None:
        self.assertEqual(_FORTRAN_TABLES["si_k444"], _FORTRAN_TABLES["si_k444_s0"])

    def test_tables(self) -> None:
        cases = {
            "si_k444": ("4 4 4", False, 1.0),
            "si_k333": ("3 3 3", False, 1.0),
            "si_k221": ("2 2 1", False, 1.0),
            "sim_k444": ("4 4 4", True, 1.0),
            "sim_k222": ("2 2 2", True, 1.0),
            "sio_k342": ("3 4 2", False, 11.25 / 10.26),
        }
        parser = ParserTests()
        parser.setUp()
        try:
            for name, (mesh, moved, stretch) in cases.items():
                table = _FORTRAN_TABLES.get(name)
                if table is None:
                    continue
                with self.subTest(case=name):
                    translation = parser.parse(mesh, "", "Use_Fortran_Routines: .true.", moved, stretch)
                    lattice = translation.problem.periodic_cell.lattice_vectors
                    group = translation.problem.space_group
                    dimensions = translation.problem.monkhorst_pack_dimensions
                    unfolded = irreducible_monkhorst_pack(
                        lattice, dimensions, (0.0, 0.0, 0.0), group.reciprocal_rotations, fold=False
                    )
                    reference = np.asarray(table, dtype=float)
                    np.testing.assert_allclose(unfolded.printed_weights, reference[:, 0], atol=5e-8)
                    np.testing.assert_allclose(unfolded.fractional, reference[:, 1:], atol=5e-5)
                    k_points, weights = kpoint_sampling(translation.problem)
                    np.testing.assert_allclose(weights, reference[:, 0] / reference[:, 0].sum(), atol=1e-7)
                    np.testing.assert_allclose(
                        k_points @ np.linalg.inv(2.0 * np.pi * np.linalg.inv(lattice).T),
                        fold_to_centered_zone(reference[:, 1:]),
                        atol=5e-5,
                    )
        finally:
            parser.doCleanups()


_WIRE_SLAB_INPUT = """
Boundary_Conditions: {boundary}
begin Cell_Shape
{cell}
end Cell_Shape
Boundary_Sphere_Radius: 5.0
Grid_Spacing: 0.6
States_Num: 2
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
{mesh}
end Monkhorst_Pack_Grid
{shift}{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.0 0.0 0.7
  0.0 0.0 -0.7
end Atom_Coord
"""


class WireSlabTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        shutil.copy(_H_POTRE, self.directory / "H_POTRE.DAT")

    def parse(self, boundary, cell, mesh, shift="", extra="", environment=None):
        path = self.directory / "parsec.in"
        path.write_text(
            _WIRE_SLAB_INPUT.format(boundary=boundary, cell=cell, mesh=mesh, shift=shift, extra=extra), encoding="utf-8"
        )
        with mock.patch.dict(os.environ, environment or {}):
            return parse_parsec_input(path)

    def test_wire_and_slab(self) -> None:
        cases = [
            ("wire", " 3.0", " 4", "begin Monkhorst_Pack_Shift\n 0.0\nend Monkhorst_Pack_Shift\n", ""),
            ("slab", " 3.0 0.0\n 0.0 3.5", " 4 3", "begin Monkhorst_Pack_Shift\n 0.0 0.0\nend Monkhorst_Pack_Shift\n", ""),
            ("slab", " 3.0 0.0\n 0.0 3.5", " 4 3", "begin Monkhorst_Pack_Shift\n 0.0 0.0\nend Monkhorst_Pack_Shift\n",
             "Use_Fortran_Routines: .false."),
        ]
        for boundary, cell, mesh, zero_block, extra in cases:
            with self.subTest(boundary=boundary, extra=extra):
                unshifted = self.parse(boundary, cell, mesh, "", extra)
                zero = self.parse(boundary, cell, mesh, zero_block, extra)
                k_ours, w_ours = kpoint_sampling(unshifted.problem)
                k_zero, w_zero = kpoint_sampling(zero.problem)
                np.testing.assert_array_equal(k_ours, k_zero)
                np.testing.assert_array_equal(w_ours, w_zero)
                self.assertEqual(unshifted.problem.monkhorst_pack_shift, zero.problem.monkhorst_pack_shift)
                self.assertEqual(unshifted.problem.scf.symmetrize_density, zero.problem.scf.symmetrize_density)
                self.assertEqual(unshifted.routines.fortran, zero.routines.fortran)
                # the k-points lie on the Gamma-centred mesh of the periodic direction(s)
                lattice = unshifted.problem.periodic_cell.lattice_vectors
                fractions = k_ours @ np.linalg.inv(2.0 * np.pi * np.linalg.inv(lattice).T)
                counts = [int(value) for value in mesh.split()] + [1] * (3 - len(mesh.split()))
                np.testing.assert_allclose(fractions * counts, np.rint(fractions * counts), atol=1e-10)
                former = self.parse(boundary, cell, mesh, "", extra, environment=_PORT)
                self.assertIsNone(former.problem.monkhorst_pack_shift)
                expected = monkhorst_pack_grid(former.problem.periodic_cell, former.problem.monkhorst_pack_dimensions)
                np.testing.assert_array_equal(kpoint_sampling(former.problem)[0], expected[0])


if __name__ == "__main__":
    unittest.main()
