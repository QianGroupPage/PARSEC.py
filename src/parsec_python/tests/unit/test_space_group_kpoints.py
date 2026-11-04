"""Fortran PARSEC's k-point machinery: space group, irreducible wedge, weights.

Covers :mod:`parsec_python.Grid.space_group` (``symmetries.f90``/``symgen.f90``),
:func:`parsec_python.Grid.kpoints.irreducible_monkhorst_pack` (``irrbz.f90``),
the ``kptgrid`` map, the PARSEC report lines and ``qpoints.dat``
(:mod:`parsec_python.Output.kpoint_output`), the per-state weights of the Fermi
level (``flevel.f90``), ``adjust_step`` with fractional translations, the
periodic ``Symmetrize_Charge_Density`` (``symm_scalar``), the input keywords
(``Monkhorst_Pack_Shift``, weighted ``Kpoint_List``) and the weighted k-point
SCF against equal-weight and full-grid runs.  Reference numbers are those of
Fortran PARSEC (``tests/BGW_interface/01-scf/parsec.out_ref`` and runs of the
WSL binary on the ``tests/orthorhombic`` and ``tests/hexagonal`` examples).
"""


# With both sets of routines: the periodic inputs and grids of this module are those of the Fortran-derived
# routines (Use_Fortran_Routines: .true., PeriodicGridSettings(fortran_routines=True)); the numbers checked
# here are those of the Fortran parity round.
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Grid.kpoints import (
    irreducible_monkhorst_pack,
    kpoint_sampling,
    kpoint_sampling_details,
    translate01,
)
from parsec_python.Grid.pbc import adjust_step, build_periodic_grid
from parsec_python.Grid.point_group import symmetrize_scalar
from parsec_python.Grid.space_group import (
    candidate_rotations,
    find_space_group,
    fortran_anint,
    grid_commensurability_problem,
    identity_space_group,
    periodic_grid_symmetry,
)
from parsec_python.Input.parsec_input import (
    ParsecInputError,
    UnsupportedParsecOptionError,
    parse_parsec_input,
)
from parsec_python.models import (
    Atom,
    EigensolverSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.Occupations import fermi_occupations, weighted_fermi_occupations
from parsec_python.Output.kpoint_output import (
    kpoint_generation_lines,
    monkhorst_pack_input_lines,
    qpoints_dat_text,
    space_group_lines,
)
from parsec_python.SCF.kpoint_weights import validated_kpoint_weights
from parsec_python.SCF.kpoints import run_scf_kpoints
from parsec_python.SCF.pbc import prepare_periodic_single_point

_AU_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"

_FCC = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]) * 10.2612
_SI_ATOMS = [Atom("Si", np.array(f) @ _FCC) for f in ([-0.125] * 3, [0.125] * 3)]
# tests/hexagonal with Lattice_Vector_Scale 3.21 ang written in bohr with the
# factor of ESDF (1e-10/0.529177e-10), so both codes have the same lattice.
_HEX = np.array(
    [[0.5, -0.866025403785, 0.0], [0.5, 0.866025403785, 0.0], [0.0, 0.0, 1.632993]]
) * (3.21 * (1.0e-10 / 0.529177e-10))
_ORTHO = np.diag([9.0, 8.0, 6.0])
_ORTHO_ATOMS = [
    Atom("Mg", np.zeros(3)),
    Atom("Al", np.array([0.0, 0.4, 0.0]) @ _ORTHO),
    Atom("Al", np.array([0.4, 0.0, 0.0]) @ _ORTHO),
]

# irrbz.f90's table for fcc Si 5x5x5, shift 0.5 (parsec.out_ref of
# tests/BGW_interface/01-scf, identical in the WSL build).
_BGW_TABLE = """\

 K-point generation:
 -------------------
    19 K points generated from Monkhorst-Pack grid :
    order =     5    5    5      shift =   0.5000  0.5000  0.5000

    I     weight        relative coord.        Cartesian coord.     E_kin (Ry)

    1   0.0160000   0.1000  0.1000  0.1000     0.061  0.061  0.061     0.011
    2   0.0480000   0.1000  0.1000  0.3000    -0.061  0.184  0.184     0.071
    3   0.0480000   0.1000  0.1000  0.5000    -0.184  0.306  0.306     0.221
    4   0.0480000   0.1000  0.1000  0.7000    -0.306  0.429  0.429     0.161
    5   0.0480000   0.1000  0.1000  0.9000    -0.429  0.551  0.551     0.041
    6   0.0480000   0.1000  0.3000  0.3000     0.061  0.061  0.306     0.101
    7   0.0960000   0.1000  0.3000  0.5000    -0.061  0.184  0.429     0.221
    8   0.0960000   0.1000  0.3000  0.7000    -0.184  0.306  0.551     0.581
    9   0.0960000   0.1000  0.3000  0.9000    -0.306  0.429  0.674     0.131
   10   0.0480000   0.1000  0.5000  0.5000     0.061  0.061  0.551     0.311
   11   0.0960000   0.1000  0.5000  0.7000    -0.061  0.184  0.674     0.341
   12   0.0480000   0.1000  0.5000  0.9000    -0.184  0.306  0.796     0.311
   13   0.0480000   0.1000  0.7000  0.7000     0.061  0.061  0.796     0.191
   14   0.0160000   0.3000  0.3000  0.3000     0.184  0.184  0.184     0.101
   15   0.0480000   0.3000  0.3000  0.5000     0.061  0.306  0.306     0.191
   16   0.0480000   0.3000  0.3000  0.7000    -0.061  0.429  0.429     0.521
   17   0.0480000   0.3000  0.5000  0.5000     0.184  0.184  0.429     0.251
   18   0.0480000   0.3000  0.5000  0.7000     0.061  0.306  0.551     0.401
   19   0.0080000   0.5000  0.5000  0.5000     0.306  0.306  0.306     0.281"""

# The symmetry block of the tests/orthorhombic example (WSL Fortran parsec.out),
# trailing blanks included.
_ORTHO_SYMMETRY_LINES = [
    "",
    " Symmetry Properties:",
    " --------------------",
    "",
    " The crystallographic system is orthorhombic with operations: ",
    "       1  2  3  4 25 26 27 28",
    "",
    " The space group of the crystal is symmorphic",
    " (this usually means that symmetry operations do not",
    " involve rotations and translations simultaneously)",
    "",
    " Operation number: ",
    "       1 28",
    "",
    "",
    "    Rotation matrices (r-lattice) and fractional translations (r-lattice)",
    "",
    "    1     1  0  0     0  1  0     0  0  1     -0.00000 -0.00000 -0.00000   E  ",
    "    2     1  0  0     0  1  0     0  0 -1     -0.00000 -0.00000 -0.00000   IC2",
    "",
    "Abelian subgroup contains only the Identity operation",
    "",
]


def _star_count(group, k, time_reversal=True):
    """Distinct members of the star of ``k`` modulo reciprocal lattice vectors."""

    seen = set()
    for matrix in group.reciprocal_rotations.astype(float):
        for inv in ((1.0, -1.0) if time_reversal else (1.0,)):
            q = np.mod(inv * (matrix @ k) + 1e-9, 1.0)
            seen.add(tuple(np.round(q * 1e6).astype(int) % 1000000))
    return len(seen)


class SpaceGroupTests(unittest.TestCase):
    def test_candidate_rotations_are_orthogonal_and_start_with_the_identity(self) -> None:
        for count in (24, 48):
            rotations = np.asarray(candidate_rotations(count))
            self.assertEqual(rotations.shape, (count, 3, 3))
            np.testing.assert_array_equal(rotations[0], np.eye(3))
            for matrix in rotations:
                np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-14)
            # Distinct operations.
            self.assertEqual(len({tuple(np.round(m, 8).ravel()) for m in rotations}), count)

    def test_diamond_has_the_full_cubic_group_with_half_translations(self) -> None:
        group = find_space_group(_FCC, _SI_ATOMS)
        self.assertEqual(group.order, 48)
        self.assertEqual(group.crystal_system, "cubic")
        self.assertTrue(group.full_holohedry)
        self.assertFalse(group.symmorphic)
        self.assertEqual(group.labels[0], "E  ")
        self.assertEqual(group.labels[24], "IE ")
        # Operation 2 of PARSEC's list: matrix and the signs it prints.
        self.assertEqual(group.rotations[1], ((0, 1, 0), (1, 0, 0), (-1, -1, -1)))
        self.assertEqual(group.labels[1], "C2 ")
        translation = group.translations[1]
        self.assertEqual(f"{translation[0]:9.5f}{translation[1]:9.5f}{translation[2]:9.5f}",
                         " -0.00000 -0.00000  0.50000")
        taus = group.commensurate_translations()
        self.assertTrue(all(abs(value - 0.5) < 1e-12 for direction in taus for value in direction))

    def test_operations_map_the_crystal_onto_itself(self) -> None:
        for lattice, atoms in ((_FCC, _SI_ATOMS), (_ORTHO, _ORTHO_ATOMS)):
            group = find_space_group(lattice, atoms)
            fractional = np.linalg.solve(lattice.T, np.array([a.position for a in atoms]).T).T
            symbols = [a.symbol for a in atoms]
            for matrix, translation in zip(group.rotation_matrices, group.translation_vectors):
                mapped = (fractional - translation) @ matrix.T
                for position, symbol in zip(mapped, symbols):
                    distances = [
                        np.max(np.abs((position - other) - np.rint(position - other)))
                        for other, other_symbol in zip(fractional, symbols)
                        if other_symbol == symbol
                    ]
                    self.assertLess(min(distances), 1e-8)

    def test_group_is_closed(self) -> None:
        group = find_space_group(_FCC, _SI_ATOMS)
        matrices = {tuple(m.ravel()) for m in group.rotation_matrices}
        for a in group.rotation_matrices:
            for b in group.rotation_matrices:
                self.assertIn(tuple((a @ b).ravel()), matrices)

    def test_hexagonal_and_orthorhombic_groups(self) -> None:
        hexagonal = find_space_group(
            _HEX,
            [
                Atom("Mg", np.array([0.333333333333, -0.333333333333, 0.25]) @ _HEX),
                Atom("Mg", np.array([-0.333333333333, 0.333333333333, -0.25]) @ _HEX),
            ],
        )
        self.assertEqual((hexagonal.order, hexagonal.crystal_system), (24, "hexagonal"))
        self.assertTrue(hexagonal.full_holohedry)
        # Fortran's signs of the half translation along c (operations 2 and 8).
        self.assertLess(hexagonal.translations[1][2], 0.0)
        self.assertGreater(hexagonal.translations[7][2], 0.0)
        ortho = find_space_group(_ORTHO, _ORTHO_ATOMS)
        self.assertEqual(ortho.order, 2)
        self.assertEqual(ortho.lattice_operations, (1, 2, 3, 4, 25, 26, 27, 28))
        self.assertEqual(ortho.candidate_numbers, (1, 28))
        self.assertTrue(ortho.symmorphic)

    def test_report_lines_match_fortran(self) -> None:
        group = find_space_group(_ORTHO, _ORTHO_ATOMS)
        self.assertEqual(space_group_lines(group), _ORTHO_SYMMETRY_LINES)
        ignored = space_group_lines(identity_space_group())
        self.assertIn(" Symmetry suppressed by the Ignore_Symmetry flag.", ignored)
        self.assertIn("skipping non-trivial symmetry operations", ignored)
        diamond = space_group_lines(find_space_group(_FCC, _SI_ATOMS))
        self.assertIn(" The point group of the system is the full        cubic group", diamond)
        self.assertIn("      37 38 39 40 41 42 43 44 45 46 47 48", diamond)

    def test_fortran_rounding(self) -> None:
        self.assertEqual(fortran_anint(0.5), 1.0)
        self.assertEqual(fortran_anint(-0.5), -1.0)
        self.assertEqual(fortran_anint(2.5), 3.0)
        self.assertEqual(str(fortran_anint(-0.3)), "-0.0")
        self.assertEqual(fortran_anint(0.49999999999999994), 0.0)


class IrreducibleWedgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.group = find_space_group(_FCC, _SI_ATOMS)
        cls.sampling = irreducible_monkhorst_pack(
            _FCC, (5, 5, 5), (0.5, 0.5, 0.5), cls.group.reciprocal_rotations
        )

    def test_fcc_table_is_fortrans(self) -> None:
        self.assertEqual("\n".join(kpoint_generation_lines(self.sampling)), _BGW_TABLE)
        self.assertEqual(self.sampling.count, 19)
        self.assertEqual(int(self.sampling.counts.sum()), 125)
        self.assertAlmostEqual(float(self.sampling.weights.sum()), 1.0, places=15)

    def test_qpoints_dat(self) -> None:
        text = qpoints_dat_text(self.sampling)
        lines = text.splitlines()
        self.assertEqual(lines[0], "begin qpoints")
        self.assertEqual(lines[1], "  0.1000000000  0.1000000000  0.1000000000  1.0  0")
        self.assertEqual(lines[-1], "end qpoints")
        self.assertEqual(len(lines), 21)
        gamma = irreducible_monkhorst_pack(_FCC, (2, 2, 2), (0, 0, 0), self.group.reciprocal_rotations)
        self.assertEqual(
            qpoints_dat_text(gamma).splitlines()[1],
            "  0.0000000000  0.0000000000  0.0000000000  1.0  1",
        )

    def test_grid_map_points_to_an_equivalent_representative(self) -> None:
        sampling = self.sampling
        full = sampling.full_grid_fractional()
        self.assertEqual(full.shape, (5, 5, 5, 3))
        np.testing.assert_array_equal(np.bincount(sampling.grid_map.ravel()), sampling.counts)
        rotations = self.group.reciprocal_rotations.astype(float)
        for index in np.ndindex(5, 5, 5):
            k = full[index]
            representative = sampling.fractional[sampling.grid_map[index]]
            images = [inv * (g @ k) for g in rotations for inv in (1.0, -1.0)]
            self.assertTrue(
                any(np.max(np.abs(q - representative - np.rint(q - representative))) < 1e-8 for q in images)
            )
        # The first member met is the representative, kept unfolded.
        np.testing.assert_array_equal(sampling.fractional[4], [0.1, 0.1, 0.9])

    def test_closed_grid_weights_are_star_sizes(self) -> None:
        sampling = irreducible_monkhorst_pack(
            _FCC, (4, 4, 4), (0.0, 0.0, 0.0), self.group.reciprocal_rotations
        )
        for k, count in zip(sampling.fractional, sampling.counts):
            self.assertEqual(_star_count(self.group, k), count)

    def test_identity_group_and_time_reversal(self) -> None:
        full = irreducible_monkhorst_pack(_ORTHO, (2, 2, 2), (0.5, 0.5, 0.5), None, time_reversal=False)
        self.assertEqual(full.count, 8)
        np.testing.assert_allclose(full.weights, np.full(8, 0.125))
        paired = irreducible_monkhorst_pack(_ORTHO, (2, 2, 2), (0.5, 0.5, 0.5), None)
        self.assertEqual(paired.count, 4)
        gamma = irreducible_monkhorst_pack(_ORTHO, (3, 1, 1), (0.0, 0.0, 0.0), None)
        np.testing.assert_array_equal(gamma.counts, [1, 2])
        ortho = find_space_group(_ORTHO, _ORTHO_ATOMS)
        reduced = irreducible_monkhorst_pack(
            _ORTHO, (2, 2, 2), (0.5, 0.5, 0.5), ortho.reciprocal_rotations
        )
        np.testing.assert_array_equal(reduced.counts, [4, 4])
        np.testing.assert_allclose(reduced.fractional, [[0.25, 0.25, 0.25], [0.25, 0.75, 0.25]])

    def test_translate01(self) -> None:
        self.assertEqual(translate01([0.0, 1.0, -1.0]), [0.0, 0.0, 0.0])
        self.assertEqual(translate01([0.25, -0.25, 1.25])[:2], [0.25, 0.75])
        self.assertEqual(translate01([1e-10, -1e-10, 0.999999999999])[:2], [0.0, 0.0])

    def test_invalid_shift_and_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            irreducible_monkhorst_pack(_FCC, (5, 5, 0), (0, 0, 0))
        with self.assertRaises(ValueError):
            irreducible_monkhorst_pack(_FCC, (5, 5, 5), (0, -0.5, 0))


class WeightedOccupationTests(unittest.TestCase):
    def test_equal_weights_reproduce_the_scalar_degeneracy(self) -> None:
        rng = np.random.default_rng(3)
        levels = np.sort(rng.normal(size=40))
        weight = 0.125
        reference = fermi_occupations(levels, 7.3, 600.0, degeneracy=2.0 * weight)
        weighted = weighted_fermi_occupations(levels, np.full(40, weight), 7.3, 600.0)
        np.testing.assert_allclose(weighted.occupations, reference.occupations, atol=1e-12)
        self.assertAlmostEqual(weighted.fermi_level, reference.fermi_level, places=10)

    def test_count_is_the_weighted_sum(self) -> None:
        levels = np.sort(np.linspace(-1.0, 1.0, 30))
        weights = np.tile([0.1, 0.3, 0.6], 10) / 10.0
        result = weighted_fermi_occupations(levels, weights, 1.5, 800.0, degeneracy=2.0)
        self.assertAlmostEqual(2.0 * float(np.dot(weights, result.occupations)), 1.5, places=10)
        self.assertAlmostEqual(result.electron_count, 1.5, places=10)
        zero = weighted_fermi_occupations(levels, weights, 1.5, 0.0, degeneracy=2.0)
        self.assertAlmostEqual(2.0 * float(np.dot(weights, zero.occupations)), 1.5, places=12)
        self.assertTrue(np.all((zero.occupations >= 0.0) & (zero.occupations <= 1.0)))

    def test_fermi_occupations_dispatches_the_weights(self) -> None:
        levels = np.sort(np.linspace(-1.0, 1.0, 12))
        weights = np.linspace(0.5, 1.5, 12) / 12.0
        direct = weighted_fermi_occupations(levels, weights, 0.9, 300.0)
        dispatched = fermi_occupations(levels, 0.9, 300.0, weights=weights)
        np.testing.assert_array_equal(direct.occupations, dispatched.occupations)

    def test_uniform_kpoint_weights_take_the_former_arithmetic(self) -> None:
        # The equal-weight path of the drivers must call fermi_occupations
        # exactly as before (degeneracy = 2*weight), so results stay bitwise.
        k_points = np.zeros((4, 3))
        _, weighting = validated_kpoint_weights(k_points, np.full(4, 0.25), "test")
        self.assertTrue(weighting.uniform)
        stacked = np.array([0.3, -0.2, 0.1, 0.5, -0.4, 0.2, 0.0, 0.7])
        occupations, fermi = weighting.occupy(stacked, 0.6, 500.0, 2.0)
        order = np.argsort(stacked, kind="stable")
        reference = fermi_occupations(stacked[order], 0.6, 500.0, degeneracy=2.0 * 0.25)
        expected = np.empty_like(stacked)
        expected[order] = reference.occupations
        np.testing.assert_array_equal(occupations, expected)
        self.assertEqual(fermi, reference.fermi_level)
        self.assertEqual(weighting.of_k(3), 0.25)
        _, unequal = validated_kpoint_weights(np.zeros((2, 3)), np.array([0.25, 0.75]), "test")
        self.assertFalse(unequal.uniform)
        self.assertEqual(unequal.of_k(1), 0.75)
        with self.assertRaises(ValueError):
            validated_kpoint_weights(np.zeros((2, 3)), np.array([0.5, 0.6]), "test")


class AdjustStepTests(unittest.TestCase):
    def test_without_translations_unchanged(self) -> None:
        self.assertEqual(adjust_step(10.0, 0.6), (16, 0.625))
        self.assertEqual(adjust_step(10.0, 0.6, translations=()), adjust_step(10.0, 0.6))

    def test_translations_make_the_step_commensurate(self) -> None:
        count, step = adjust_step(10.0, 0.6, translations=(1.0 / 3.0,))
        self.assertEqual(count % 6, 0)
        self.assertGreaterEqual(step, 0.6)
        self.assertEqual(count, 12)
        count, _ = adjust_step(10.0, 0.6, translations=(0.25, 0.5))
        self.assertEqual(count % 4, 0)
        self.assertEqual(count, 16)


class PeriodicSymmetrizationTests(unittest.TestCase):
    def test_general_grid_permutations(self) -> None:
        group = find_space_group(_FCC, _SI_ATOMS)
        settings = PeriodicGridSettings(
            fortran_routines=True,
            spacing=0.7,
            expansion_order=4,
            box_lengths=tuple(np.linalg.norm(_FCC, axis=1)),
            fractional_translations=group.commensurate_translations(),
        )
        grid = build_periodic_grid(PeriodicCell(lattice_vectors=_FCC), settings)
        mapping = periodic_grid_symmetry(grid, group)
        self.assertEqual(mapping.order, 48)
        self.assertEqual(mapping.reduced_order, 1)
        for images in mapping.images:
            self.assertEqual(np.unique(images).size, grid.size)
        rng = np.random.default_rng(5)
        field = rng.random(grid.size)
        symmetric = symmetrize_scalar(field, mapping)
        np.testing.assert_allclose(symmetrize_scalar(symmetric, mapping), symmetric, atol=1e-14)
        for images in mapping.images:
            np.testing.assert_allclose(symmetric[images], symmetric, atol=1e-14)
        self.assertAlmostEqual(float(symmetric.sum()), float(field.sum()), places=9)

    def test_orthorhombic_grid_must_be_commensurate(self) -> None:
        cube = 10.26 * np.eye(3)
        fractions = [
            (0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0),
            (0.25, 0.25, 0.25), (0.25, 0.75, 0.75), (0.75, 0.25, 0.75), (0.75, 0.75, 0.25),
        ]
        group = find_space_group(cube, [Atom("Si", np.array(f) @ cube) for f in fractions])
        self.assertEqual(group.order, 48)
        self.assertIsNone(grid_commensurability_problem(group, (16, 16, 16), (0, 0, 0)))
        problem = grid_commensurability_problem(group, (18, 18, 18), (0, 0, 0))
        self.assertIn("does not map the grid 18x18x18", problem)


_PERIODIC_INPUT = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk

begin Cell_Shape
   8.0  8.0  8.0
end Cell_Shape

Kpoint_Method: mp
begin Monkhorst_Pack_Grid
2 2 2
end Monkhorst_Pack_Grid
{shift}
Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 10
Fermi_Temp: 500.0
Max_Iter: 30
Convergence_Criterion: 5.0e-3
Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

Atom_Types_Num: 1
Atom_Type: Au
Local_Component: s
Begin atom_coord
   4.0 4.0 4.0
End atom_coord
{extra}
"""

_SHIFT_BLOCK = "begin Monkhorst_Pack_Shift\n0.0 0.0 0.0\nend Monkhorst_Pack_Shift\n"


class InputTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        shutil.copy(_AU_POTRE, self.directory / "Au_POTRE.DAT")

    def _parse(self, shift: str = _SHIFT_BLOCK, extra: str = "", text: str | None = None):
        path = self.directory / "parsec.in"
        path.write_text(text or _PERIODIC_INPUT.format(shift=shift, extra=extra), encoding="utf-8")
        return parse_parsec_input(path)

    def test_shift_block_takes_fortrans_path(self) -> None:
        translation = self._parse()
        problem = translation.problem
        self.assertEqual(problem.monkhorst_pack_shift, (0.0, 0.0, 0.0))
        self.assertEqual(problem.space_group.order, 48)
        self.assertTrue(problem.scf.symmetrize_density)
        details = kpoint_sampling_details(problem)
        np.testing.assert_array_equal(details.counts, [1, 3, 3, 1])
        k_points, weights = kpoint_sampling(problem)
        np.testing.assert_allclose(weights, [0.125, 0.375, 0.375, 0.125])
        self.assertEqual(k_points.shape, (4, 3))
        self.assertIn("monkhorst_pack_shift", repr(problem))

    def test_without_shift_nothing_changes(self) -> None:
        problem = self._parse(shift="").problem
        self.assertIsNone(problem.monkhorst_pack_shift)
        self.assertIsNone(problem.space_group)
        self.assertFalse(problem.scf.symmetrize_density)
        self.assertNotIn("space_group", repr(problem))
        self.assertNotIn("monkhorst_pack_shift", repr(problem))
        self.assertNotIn("fractional_translations", repr(problem.grid))
        self.assertNotIn("symmetrize_density", repr(problem.scf))
        _, weights = kpoint_sampling(problem)
        np.testing.assert_allclose(weights, np.full(8, 0.125))

    def test_symmetrization_switches_and_ignore_symmetry(self) -> None:
        off = self._parse(extra="Symmetrize_Charge_Density: .false.\n")
        self.assertFalse(off.problem.scf.symmetrize_density)
        self.assertTrue(any("Symmetrize_Charge_Density: false" in w for w in off.warnings))
        ignored = self._parse(extra="Ignore_Symmetry: .true.\n").problem
        self.assertTrue(ignored.space_group.ignored)
        # Every point of the Gamma-centred 2x2x2 grid is its own -k.
        np.testing.assert_array_equal(kpoint_sampling_details(ignored).counts, np.ones(8))
        gamma = self._parse(
            shift="", text=_PERIODIC_INPUT.format(shift="", extra="Symmetrize_Charge_Density: .true.\n")
            .replace("Kpoint_Method: mp", "Kpoint_Method: none")
            .replace("begin Monkhorst_Pack_Grid\n2 2 2\nend Monkhorst_Pack_Grid\n", "")
        ).problem
        self.assertTrue(gamma.scf.symmetrize_density)
        self.assertEqual(gamma.space_group.order, 48)

    def test_shift_errors_and_warnings(self) -> None:
        with self.assertRaises(ParsecInputError):
            self._parse(shift="begin Monkhorst_Pack_Shift\n0.5 0.5\nend Monkhorst_Pack_Shift\n")
        with self.assertRaises(ParsecInputError):
            self._parse(shift="begin Monkhorst_Pack_Shift\n0.5 -0.5 0\nend Monkhorst_Pack_Shift\n")
        with self.assertRaises(ParsecInputError):
            self._parse(
                shift="begin Monkhorst_Pack_Shift\n0.5 0.5 0.5\n0 0 0\nend Monkhorst_Pack_Shift\n"
            )
        text = (
            _PERIODIC_INPUT.format(shift=_SHIFT_BLOCK, extra="")
            .replace("Kpoint_Method: mp", "Kpoint_Method: none")
            .replace("begin Monkhorst_Pack_Grid\n2 2 2\nend Monkhorst_Pack_Grid\n", "")
        )
        translation = self._parse(text=text)
        self.assertIsNone(translation.problem.monkhorst_pack_shift)
        self.assertTrue(any("Monkhorst_Pack_Shift is read only" in w for w in translation.warnings))

    def test_weighted_kpoint_list_is_normalized(self) -> None:
        text = (
            _PERIODIC_INPUT.format(shift="", extra="")
            .replace(
                "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 2\nend Monkhorst_Pack_Grid\n",
                "Kpoint_Method: manual\nbegin Kpoint_List\n0 0 0 1\n0.2 0 0 3\nend Kpoint_List\n",
            )
        )
        problem = self._parse(text=text).problem
        np.testing.assert_allclose(problem.kpoint_weights, [0.25, 0.75])
        self.assertFalse(problem.scf.symmetrize_density)
        self.assertIsNone(problem.space_group)

    def test_forces_of_a_reduced_kpoint_set_are_accepted(self) -> None:
        # The forces of an irreducible wedge are completed over the space
        # group (tests/unit/test_space_group_forces.py).
        translation = self._parse(extra="Skip_force: .false.\n")
        self.assertTrue(translation.compute_forces)
        self.assertEqual(translation.problem.space_group.order, 48)
        self.assertFalse(any("energies only" in w for w in translation.warnings))
        relaxation = self._parse(extra="Minimization: bfgs\n")
        self.assertEqual(relaxation.relaxation.method, "bfgs")
        self.assertTrue(self._parse(shift="", extra="Skip_force: .false.\n").compute_forces)

    def test_noncommensurate_orthorhombic_grid_is_refused(self) -> None:
        text = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  10.26  10.26  10.26
end Cell_Shape
Coordinate_Unit: lattice_vectors
Grid_Spacing: {spacing}
Expansion_Order: 4
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
2 2 2
end Monkhorst_Pack_Grid
begin Monkhorst_Pack_Shift
0.5 0.5 0.5
end Monkhorst_Pack_Shift
States_Num: 20
Atom_Types_Num: 1
Atom_Type: Au
Local_Component: s
begin Atom_Coord
  0.00  0.00  0.00
  0.00  0.50  0.50
  0.50  0.00  0.50
  0.50  0.50  0.00
  0.25  0.25  0.25
  0.25  0.75  0.75
  0.75  0.25  0.75
  0.75  0.75  0.25
end Atom_Coord
"""
        with self.assertRaises(UnsupportedParsecOptionError) as caught:
            self._parse(text=text.format(spacing=0.57))
        self.assertIn("Symmetrize_Charge_Density", str(caught.exception))
        problem = self._parse(text=text.format(spacing=0.64)).problem
        self.assertEqual(problem.space_group.order, 48)
        problem = self._parse(
            text=text.format(spacing=0.57) + "Symmetrize_Charge_Density: .false.\n"
        ).problem
        self.assertFalse(problem.scf.symmetrize_density)

    def test_noncollinear_symmetrization_follows_parsec(self) -> None:
        # rho and |m| averaged as scalars, the direction of m kept (newrho.F90).
        translation = self._parse(extra="Non_Collinear_magnetism: .true.\n")
        self.assertTrue(translation.problem.scf.symmetrize_density)
        self.assertTrue(any("|m| as scalars" in w for w in translation.warnings))
        problem = self._parse(
            extra="Non_Collinear_magnetism: .true.\nIgnore_Symmetry: .true.\n"
        ).problem
        self.assertFalse(problem.scf.symmetrize_density)
        self.assertFalse(kpoint_sampling_details(problem).time_reversal)

    def test_header_lines(self) -> None:
        self.assertEqual(
            monkhorst_pack_input_lines((5, 5, 5), (0.5, 0.5, 0.5)),
            [
                "  kpoints selected by Monkhort-Pack algorithm",
                "  Monkhorst-Pack grid dimensions specified",
                "   5 X  5 X  5",
                "  Monkhorst-Pack shift specified",
                "   0.500 X  0.500 X  0.500",
            ],
        )


def _au_problem(**fields) -> SinglePointInput:
    cell = PeriodicCell(lattice_vectors=8.0 * np.eye(3))
    return SinglePointInput(
        atoms=[Atom("Au", [4.0, 4.0, 4.0])],
        pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0)},
        grid=PeriodicGridSettings(fortran_routines=True, spacing=0.6, expansion_order=6),
        periodic_cell=cell,
        scf=SCFSettings(
            max_iterations=1,
            number_of_states=10,
            fermi_temperature_kelvin=2000.0,
            symmetrize_density=fields.pop("symmetrize", False),
        ),
        eigensolver=EigensolverSettings(method="chebdav", tolerance=1.0e-9),
        **fields,
    )


class WeightedSCFTests(unittest.TestCase):
    """One SCF iteration from the same start: equal physics, other k-point sets."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.plain = prepare_periodic_single_point(_au_problem())

    def test_duplicated_points_with_split_weights(self) -> None:
        k = np.array([[0.0, 0.0, 0.0], [np.pi / 8.0, 0.0, 0.0]])
        equal = run_scf_kpoints(self.plain, k, np.array([0.5, 0.5]))
        split = run_scf_kpoints(
            self.plain, k[[0, 0, 1]], np.array([0.25, 0.25, 0.5])
        )
        self.assertAlmostEqual(split.energies.total, equal.energies.total, delta=1e-8)
        self.assertAlmostEqual(split.fermi_level, equal.fermi_level, delta=1e-8)
        np.testing.assert_allclose(split.density, equal.density, atol=1e-9)

    def test_irreducible_wedge_reproduces_the_full_grid(self) -> None:
        group = find_space_group(8.0 * np.eye(3), [Atom("Au", np.array([4.0, 4.0, 4.0]))])
        reduced_problem = _au_problem(
            symmetrize=True,
            monkhorst_pack_dimensions=(2, 2, 2),
            monkhorst_pack_shift=(0.0, 0.0, 0.0),
            space_group=group,
        )
        reduced_system = prepare_periodic_single_point(reduced_problem)
        k_reduced, w_reduced = kpoint_sampling(reduced_problem)
        self.assertEqual(len(w_reduced), 4)
        reduced = run_scf_kpoints(reduced_system, k_reduced, w_reduced)
        self.assertEqual(len(reduced.history[-1].density_symmetrization), 1)
        # The full grid, each point as a rotated image of its representative.
        details = kpoint_sampling_details(reduced_problem)
        rotations = group.reciprocal_rotations.astype(float)
        stars = []
        for k, count in zip(details.fractional, details.counts):
            members = {}
            for g in rotations:
                for inv in (1.0, -1.0):
                    q = inv * (g @ k)
                    members.setdefault(tuple(np.round(np.mod(q + 1e-9, 1.0), 6)), q)
            self.assertEqual(len(members), count)
            stars.extend(members.values())
        k_full = np.asarray(stars) @ (2.0 * np.pi * np.linalg.inv(8.0 * np.eye(3)).T)
        full = run_scf_kpoints(self.plain, k_full, np.full(len(stars), 1.0 / len(stars)))
        self.assertAlmostEqual(reduced.energies.total, full.energies.total, delta=1e-8)
        self.assertAlmostEqual(reduced.fermi_level, full.fermi_level, delta=1e-7)
        # Converged eigenvalues (1e-9 Ry), eigenvectors to about 1e-7.
        np.testing.assert_allclose(reduced.density, full.density, atol=2e-6)


class PeriodicGammaSymmetrizationTests(unittest.TestCase):
    """The Gamma-point loops of a cell apply ``Symmetrize_Charge_Density`` too."""

    def test_scalar_and_spin_gamma_loops(self) -> None:
        from dataclasses import replace

        from parsec_python.SCF.single_point import run_scf
        from parsec_python.SCF.spin_polarized import run_scf_spin_polarized

        group = find_space_group(8.0 * np.eye(3), [Atom("Au", np.array([4.0, 4.0, 4.0]))])
        problem = _au_problem(symmetrize=True, space_group=group)
        system = prepare_periodic_single_point(problem)
        self.assertEqual(system.density_symmetry.order, 48)
        result = run_scf(system)
        reports = result.history[-1].density_symmetrization
        self.assertEqual(len(reports), 1)
        self.assertAlmostEqual(reports[0].charge_after, reports[0].charge_before, places=10)
        density = result.density
        for images in system.density_symmetry.images:
            np.testing.assert_allclose(density[images], density, atol=1e-14)
        spin_problem = replace(problem, scf=replace(problem.scf, spin_polarized=True))
        spin_system = prepare_periodic_single_point(spin_problem)
        records = []
        run_scf_spin_polarized(spin_system, callback=records.append)
        self.assertEqual(len(records[-1].density_symmetrization), 3)


class CliTests(unittest.TestCase):
    def test_report_and_qpoints_file(self) -> None:
        from parsec_python.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shutil.copy(_AU_POTRE, directory / "Au_POTRE.DAT")
            text = _PERIODIC_INPUT.format(shift=_SHIFT_BLOCK, extra="").replace(
                "Max_Iter: 30", "Max_Iter: 2"
            )
            (directory / "parsec.in").write_text(text, encoding="utf-8")
            code = cli_main([str(directory / "parsec.in"), "--quiet", "--no-archive"])
            self.assertIn(code, (0, 3))
            log = (directory / "parsec.out").read_text(encoding="utf-8")
            for line in (
                " Symmetry Properties:",
                " The point group of the system is the full        cubic group",
                "     4 K points generated from Monkhorst-Pack grid :",
                "    1   0.1250000   0.0000  0.0000  0.0000     0.000  0.000  0.000     0.000",
                "  Symmetrizing charge isp =            1",
                "  Monkhorst-Pack shift specified",
                " WAVEFUNCTIONS ARE COMPLEX!",
            ):
                self.assertIn(line, log)
            qpoints = (directory / "qpoints.dat").read_bytes()
            self.assertTrue(qpoints.startswith(b"begin qpoints\n"))
            self.assertEqual(qpoints.count(b"\n"), 6)

    def test_accelerated_cli_routes_to_the_reference_solver(self) -> None:
        from parsec_python.acceleration.cli import _reference_only_features

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shutil.copy(_AU_POTRE, directory / "Au_POTRE.DAT")
            (directory / "parsec.in").write_text(
                _PERIODIC_INPUT.format(shift=_SHIFT_BLOCK, extra=""), encoding="utf-8"
            )
            translation = parse_parsec_input(directory / "parsec.in")
            features = _reference_only_features(translation, accelerated_backend_selected=False)
            self.assertTrue(any("k-point sampling" in item for item in features))
            self.assertTrue(any("Symmetrize_Charge_Density" in item for item in features))


if __name__ == "__main__":
    unittest.main()
