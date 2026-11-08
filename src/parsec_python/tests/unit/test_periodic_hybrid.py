"""Self-consistent hybrid functionals of periodic cells (host path, parsec_python.SCF.hybrid_periodic).

Independent checks, exact to round-off unless stated:

* the pair potentials of the Coulomb kernel at ``q = 0`` are the spectral Hartree
  potential of the cell (:func:`parsec_python.Hartree.gspace.solve_hartree_g_space`);
* the exchange of real orbitals at the Gamma point through the cluster code
  (``apply_exchange_to_orbitals`` with the FFT pair solver) equals the Bloch code at
  ``k = 0``;
* supercell: the exchange energy and ``V_x u`` of a cell with the mesh ``{0, b1/2}``
  equal those of the 2x1x1 supercell at the Gamma point for the unfolded orbitals
  ``e^{ik.r} u_k`` (band-limited orbitals: no Nyquist component);
* time reversal: the reduced mesh with ``u_{-k} = conj(u_k)`` gives the energy and
  ``V_x u`` of the full mesh;
* ACE: ``V_ACE U = V_x U`` at every k-point;
* the Fock loop at the Gamma point on real orbitals and on the 1x1x1 mesh of Bloch
  factors reach the same energy (H2 in a box), and two equal spin channels the
  unpolarized one;
* the time-reversal reduction of a hybrid's k-point sampling, the mesh counts of a
  written-out list and the refusals of the input.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from parsec_python.Grid.kpoints import time_reversal_reduced
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Hartree.gspace import solve_hartree_g_space
from parsec_python.models import (
    Atom,
    EigensolverSettings,
    ExactExchangeSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.V_xc.bloch_exchange import (
    BlochOrbitalSet,
    GridBox,
    PeriodicPairPoisson,
    PeriodicPairPotentials,
    apply_bloch_exchange,
    bloch_orbital_set,
    compress_bloch_exchange,
)
from parsec_python.V_xc.exchange_operator import apply_exchange_to_orbitals


LIBRARY = Path(__file__).resolve().parents[3] / "pp_generation"
HYDROGEN = LIBRARY / "NCPPs" / "PBE" / "H" / "PARSEC" / "H_POTRE.DAT"


def _grid(lengths, spacing=0.5):
    cell = PeriodicCell(lattice_vectors=np.diag(np.asarray(lengths, dtype=float)))
    return build_periodic_grid(cell, PeriodicGridSettings(spacing=spacing, expansion_order=4))


class StubPeriodicExchangeKernel:
    """``K(q + G)`` of ``alpha 2/r + beta 2 erfc(omega r)/r`` with the ``q + G = 0`` element zero (Ry bohr**2).

    The plain kernel of the algebraic checks below (any q, no k-mesh, no Nyquist averaging).

    ``alpha 8 pi/k**2 + beta (8 pi/k**2)(1 - exp(-k**2/(4 omega**2)))`` at
    ``k = |q + G|``; at ``k = 0`` the erfc part has the finite limit
    ``beta 2 pi/omega**2``, the Coulomb part is left out.  The signature of
    ``parsec_python.Hartree.exchange_kernel.PeriodicExchangeKernel``
    (DESIGN.md of the round, interface 1); it stands in for it until that
    module exists and has no treatment of the singularity (the energy then
    converges slowly with the k-mesh) and no truncation of a slab.
    ``lattice`` rows are the box lattice (bohr), ``shape`` the grid box and
    the FFT box; ``k_mesh``, ``periodic_axes`` and ``divergence`` are
    accepted and not used.
    """

    def __init__(
        self,
        lattice,
        shape,
        k_mesh=None,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        omega: float | None = None,
        periodic_axes=(0, 1, 2),
        divergence="none",
        x_gamma_extrapolation: bool = False,
    ) -> None:
        self.lattice = np.asarray(lattice, dtype=np.float64)
        self.shape = tuple(int(value) for value in shape)
        self.box_shape = self.shape
        self.k_mesh = k_mesh
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.omega = None if omega is None else float(omega)
        if self.beta != 0.0 and (self.omega is None or self.omega <= 0.0):
            raise ValueError("an erfc part needs a positive omega")
        self.periodic_axes = tuple(periodic_axes)
        self.divergence = divergence
        reciprocal = 2.0 * np.pi * np.linalg.inv(self.lattice).T
        indices = np.meshgrid(*(np.fft.fftfreq(n, d=1.0 / n) for n in self.shape), indexing="ij")
        self._g = sum(index[..., None] * reciprocal[axis] for axis, index in enumerate(indices))
        self.description = "stub kernel 8 pi/|q+G|^2, q+G = 0 element zero (no singularity treatment)"

    def values(self, q, xp=np) -> np.ndarray:
        """``K(q + G)`` on the FFT box (Ry bohr**2); ``xp`` is accepted for the interface (NumPy only)."""

        vectors = self._g + np.asarray(q, dtype=np.float64)
        squared = np.einsum("...i,...i->...", vectors, vectors)
        zero = squared <= 1.0e-24
        safe = np.where(zero, 1.0, squared)
        values = np.zeros(self.shape)
        if self.alpha != 0.0:
            values += self.alpha * 8.0 * np.pi / safe
        if self.beta != 0.0:
            values += self.beta * 8.0 * np.pi * (-np.expm1(-safe / (4.0 * self.omega * self.omega))) / safe
        values[zero] = 0.0 if self.beta == 0.0 else self.beta * 2.0 * np.pi / (self.omega * self.omega)
        return values


def _potentials(grid, alpha=1.0, beta=0.0, omega=None):
    box = GridBox(grid)
    kernel = StubPeriodicExchangeKernel(box.lattice, box.shape, (1, 1, 1), alpha=alpha, beta=beta, omega=omega)
    return PeriodicPairPotentials(box, kernel)


def _smooth_orbitals(grid, count, seed, real=False):
    """``count`` orthonormal grid vectors with Fourier components ``|m_i| <= 1`` only."""

    generator = np.random.default_rng(seed)
    shape = grid.shape
    columns = []
    for _ in range(count):
        coefficients = np.zeros(shape, dtype=np.complex128)
        for m in np.ndindex(3, 3, 3):
            index = tuple(value - 1 for value in m)
            coefficients[index] = generator.normal() + 1j * generator.normal()
        values = np.fft.ifftn(coefficients).reshape(-1)
        columns.append(values.real if real else values)
    block = np.column_stack(columns)
    orthonormal, _ = np.linalg.qr(block)
    return orthonormal


class PairPotentialTests(unittest.TestCase):
    def test_coulomb_kernel_at_gamma_is_the_spectral_hartree_potential(self) -> None:
        grid = _grid((4.0, 4.5, 5.0))
        density = np.random.default_rng(3).normal(size=grid.size)
        potentials = _potentials(grid)
        solved = potentials.solve_real(density[:, None])[:, 0]
        np.testing.assert_allclose(solved, solve_hartree_g_space(density, grid), rtol=0.0, atol=1.0e-12)
        complex_solved = potentials.solve(density[:, None].astype(np.complex128), np.zeros(3))[:, 0]
        np.testing.assert_allclose(complex_solved.real, solved, rtol=0.0, atol=1.0e-12)
        self.assertLess(float(np.max(np.abs(complex_solved.imag))), 1.0e-12)

    def test_real_gamma_exchange_equals_the_bloch_exchange_at_k_zero(self) -> None:
        grid = _grid((4.0, 4.0, 4.0))
        orbitals = _smooth_orbitals(grid, 4, seed=1, real=True)
        occupations = np.array([1.0, 1.0, 0.3, 0.0])
        potentials = _potentials(grid, alpha=1.0, beta=0.5, omega=0.4)
        real = apply_exchange_to_orbitals(
            orbitals, occupations, grid.volume_element, PeriodicPairPoisson(potentials), spin_weight=2.0
        )
        box = GridBox(grid)
        bloch = apply_bloch_exchange(
            bloch_orbital_set(np.zeros((1, 3)), np.ones(1), [orbitals], [occupations], 2.0, box), potentials
        )
        self.assertAlmostEqual(bloch.energy, real.energy, delta=1.0e-12 * abs(real.energy))
        np.testing.assert_allclose(bloch.applied[0].real, real.applied, rtol=0.0, atol=1.0e-12)


class SlabBoxTests(unittest.TestCase):
    """The box of the Fortran-derived slab grid: its active points, the open axis and the padded kernel."""

    def test_confined_slab_grid_box(self) -> None:
        from parsec_python.Grid.confined import build_confined_periodic_grid, confined_lattice_vectors

        vectors = np.array([[1.0, 0.0, 0.0], [0.5, 0.8660254037844386, 0.0]]) * 4.73
        cell = PeriodicCell(confined_lattice_vectors(vectors))
        settings = PeriodicGridSettings(
            spacing=0.5, shift=(0.0, 0.0, 0.5), expansion_order=6, periodicity=2, confinement_radius=4.0
        )
        grid = build_confined_periodic_grid(cell, settings)
        box = GridBox(grid)
        self.assertEqual(box.periodic_axes, (0, 1))
        np.testing.assert_allclose(box.lattice[2], [0.0, 0.0, grid.points_per_axis[2] * 0.5], atol=1.0e-12)
        values = np.random.default_rng(2).normal(size=(grid.size, 3))
        np.testing.assert_array_equal(box.from_box(box.to_box(values)), values)
        # the default slab kernel (padded) doubles the planes; the potentials of a density come back on the grid
        from parsec_python.V_xc.bloch_exchange import periodic_exchange_kernel

        kernel = periodic_exchange_kernel(box, (3, 3, 1), alpha=1.0, beta=0.0, omega=None)
        potentials = PeriodicPairPotentials(box, kernel)
        self.assertTrue(potentials.padded)
        self.assertEqual(potentials.fft_shape[2], 2 * grid.points_per_axis[2])
        solved = potentials.solve_real(np.abs(values[:, :2]))
        self.assertEqual(solved.shape, (grid.size, 2))
        self.assertTrue(np.all(np.isfinite(solved)))


class BlochExchangeTests(unittest.TestCase):
    def test_cell_with_mesh_equals_the_supercell_at_gamma(self) -> None:
        cell_grid = _grid((4.0, 4.0, 4.0))
        super_grid = _grid((8.0, 4.0, 4.0))
        b1 = 2.0 * np.pi / 4.0
        k_points = np.array([[0.0, 0.0, 0.0], [0.5 * b1, 0.0, 0.0]])
        blocks = [_smooth_orbitals(cell_grid, 3, seed=10 + t) for t in range(2)]
        occupations = [np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.5, 0.0])]
        cell_set = bloch_orbital_set(k_points, np.array([0.5, 0.5]), blocks, occupations, 2.0, GridBox(cell_grid))
        self.assertFalse(np.any(cell_set.partners))
        potentials = _potentials(cell_grid)
        cell = apply_bloch_exchange(cell_set, potentials, batch_size=5)

        def unfold(values, k_point):
            tiled = np.concatenate([values.reshape(cell_grid.shape)] * 2, axis=0).reshape(-1)
            return np.exp(1j * super_grid.coordinates @ k_point) * tiled / np.sqrt(2.0)

        unfolded = np.column_stack(
            [unfold(block[:, n], k) for block, k in zip(blocks, k_points) for n in range(block.shape[1])]
        )
        super_set = BlochOrbitalSet(
            np.zeros((1, 3)), np.ones(1), (unfolded,), (np.concatenate(occupations),), 2.0, np.array([False]), 1
        )
        supercell = apply_bloch_exchange(super_set, _potentials(super_grid))
        self.assertAlmostEqual(supercell.energy, 2.0 * cell.energy, delta=1.0e-12 * abs(cell.energy))
        for t, (block, k_point) in enumerate(zip(blocks, k_points)):
            for n in range(block.shape[1]):
                expected = unfold(cell.applied[t][:, n], k_point)
                np.testing.assert_allclose(supercell.applied[0][:, 3 * t + n], expected, rtol=0.0, atol=1.0e-12)

    def test_time_reversal_reduced_mesh_equals_the_full_mesh(self) -> None:
        grid = _grid((4.0, 4.0, 4.0))
        b1 = 2.0 * np.pi / 4.0
        minus = _smooth_orbitals(grid, 3, seed=21)
        # k = 0 is its own partner: its orbitals span a set closed under conjugation (real here)
        zero = _smooth_orbitals(grid, 3, seed=22, real=True)
        f_minus, f_zero = np.array([1.0, 0.8, 0.0]), np.array([1.0, 1.0, 0.1])
        k_minus = np.array([-b1 / 3.0, 0.0, 0.0])
        potentials = _potentials(grid, alpha=0.3, beta=0.7, omega=0.5)
        reduced_set = bloch_orbital_set(
            np.array([k_minus, np.zeros(3)]), np.array([2.0 / 3.0, 1.0 / 3.0]), [minus, zero], [f_minus, f_zero], 2.0, GridBox(grid)
        )
        self.assertEqual(reduced_set.partners.tolist(), [True, False])
        self.assertEqual(reduced_set.mesh_size, 3)
        reduced = apply_bloch_exchange(reduced_set, potentials)
        full_set = BlochOrbitalSet(
            np.array([k_minus, np.zeros(3), -k_minus]),
            np.full(3, 1.0 / 3.0),
            (minus, zero, np.conj(minus)),
            (f_minus, f_zero, f_minus),
            2.0,
            np.array([False, False, False]),
            3,
        )
        full = apply_bloch_exchange(full_set, potentials)
        self.assertAlmostEqual(reduced.energy, full.energy, delta=1.0e-12 * abs(full.energy))
        np.testing.assert_allclose(reduced.applied[0], full.applied[0], rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(reduced.applied[1], full.applied[1], rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(np.conj(reduced.applied[0]), full.applied[2], rtol=0.0, atol=1.0e-12)

    def test_pair_symmetry_solves_fewer_pairs_with_the_same_result(self) -> None:
        from parsec_python.Hartree.exchange_kernel import PeriodicExchangeKernel

        grid = _grid((4.0, 4.0, 4.0))
        box = GridBox(grid)
        b = 2.0 * np.pi / 4.0
        # the time-reversal representatives of the 3x2x1 mesh (2i-n-1)/(2n): partners and a self-partner set
        k_points, weights = time_reversal_reduced(
            np.array([[(2 * i - 4) / 6.0 * b, (2 * j - 3) / 4.0 * b, 0.0] for i in range(1, 4) for j in range(1, 3)]),
            np.full(6, 1.0 / 6.0),
            np.diag([4.0, 4.0, 4.0]),
        )
        blocks = [_smooth_orbitals(grid, 4, seed=40 + t) for t in range(len(k_points))]
        occupations = [np.array([1.0, 1.0, 0.4, 0.0])] * len(k_points)
        orbital_set = bloch_orbital_set(k_points, weights, blocks, occupations, 2.0, box)
        self.assertTrue(np.any(orbital_set.partners))
        kernel = PeriodicExchangeKernel(box.lattice, box.shape, (3, 2, 1), alpha=0.6, beta=0.4, omega=0.5)
        potentials = PeriodicPairPotentials(box, kernel)
        plain = apply_bloch_exchange(orbital_set, potentials, pair_symmetry=False, batch_size=7)
        paired = apply_bloch_exchange(orbital_set, potentials, batch_size=7)
        self.assertLess(paired.pair_solves, plain.pair_solves)
        self.assertAlmostEqual(paired.energy, plain.energy, delta=1.0e-13 * abs(plain.energy))
        for left, right in zip(paired.applied, plain.applied):
            np.testing.assert_allclose(left, right, rtol=0.0, atol=1.0e-13)
        # occupied columns first, the rest later (the last outer iteration and the next operator)
        partial = apply_bloch_exchange(orbital_set, potentials, columns="occupied")
        completed = apply_bloch_exchange(orbital_set, potentials, known=partial)
        for left, right in zip(completed.applied, plain.applied):
            np.testing.assert_allclose(left, right, rtol=0.0, atol=1.0e-13)

    def test_ace_reproduces_the_exchange_on_its_orbitals(self) -> None:
        grid = _grid((4.0, 4.0, 4.0))
        b1 = 2.0 * np.pi / 4.0
        blocks = [_smooth_orbitals(grid, 4, seed=30 + t) for t in range(2)]
        occupations = [np.array([1.0, 1.0, 0.0, 0.0])] * 2
        orbital_set = bloch_orbital_set(
            np.array([[-b1 / 4.0, 0.0, 0.0], [b1 / 4.0, 0.0, b1 / 4.0]]),
            np.array([0.5, 0.5]),
            blocks,
            occupations,
            2.0,
            GridBox(grid),
        )
        application = apply_bloch_exchange(orbital_set, _potentials(grid))
        partial = apply_bloch_exchange(orbital_set, _potentials(grid), columns="occupied")
        completed = apply_bloch_exchange(orbital_set, _potentials(grid), known=partial)
        self.assertAlmostEqual(partial.energy, application.energy, delta=1.0e-13 * abs(application.energy))
        for block, applied, done in zip(blocks, application.applied, completed.applied):
            np.testing.assert_allclose(done, applied, rtol=0.0, atol=1.0e-13)
            operator = compress_bloch_exchange(block, applied, fraction=0.25)
            np.testing.assert_allclose(operator.apply(block, scaled=False), applied, rtol=0.0, atol=1.0e-12)
            self.assertLess(operator.asymmetry, 1.0e-12)

    def test_a_mesh_that_is_not_time_reversal_reduced_is_refused(self) -> None:
        grid = _grid((4.0, 4.0, 4.0))
        b1 = 2.0 * np.pi / 4.0
        with self.assertRaises(NotImplementedError):
            bloch_orbital_set(
                np.array([[b1 / 3.0, 0.0, 0.0], [-b1 / 3.0, 0.0, 0.0]]),
                np.array([0.5, 0.5]),
                [np.zeros((grid.size, 1))] * 2,
                [np.zeros(1)] * 2,
                2.0,
                GridBox(grid),
            )


class HermitianDIISTests(unittest.TestCase):
    """The outer DIIS of Bloch orbitals: overlaps ``Y^H Y`` and natural orbitals of a complex density matrix."""

    def test_natural_orbitals_of_an_extrapolated_bloch_density_matrix(self) -> None:
        from parsec_python.SCF.fock_diis import OuterDIIS, OutputSet
        from parsec_python.SCF.hybrid_periodic import BlochOccupiedBlock

        generator = np.random.default_rng(5)
        rows, count = 40, 3
        first, _ = np.linalg.qr(generator.normal(size=(rows, count)) + 1j * generator.normal(size=(rows, count)))
        # a small unitary rotation of the space: the second output
        generator_matrix = 0.01 * (generator.normal(size=(rows, rows)) + 1j * generator.normal(size=(rows, rows)))
        hermitian = 0.5 * (generator_matrix + generator_matrix.conj().T)
        values, vectors = np.linalg.eigh(hermitian)
        rotation = (vectors * np.exp(1j * values)) @ vectors.conj().T
        second = rotation @ first
        occupations = np.ones(count)
        diis = OuterDIIS(enabled=True, tolerance=1.0e-6)
        for step, block in ((1, first), (2, second)):
            diis.append(OutputSet(step, [{7: BlochOccupiedBlock(block, occupations)}], {step - 1: 1.0}))
        # Tr(Y_a Y_b) = sum |(Y_a^H Y_b)_ij|^2 with unit occupations
        overlap = first.conj().T @ second
        self.assertAlmostEqual(diis.traces[(1, 2)], float(np.sum(np.abs(overlap) ** 2)), places=12)
        channels = diis.natural_orbitals({1: 1.5, 2: -0.5})
        self.assertIsNotNone(channels)
        phi, natural = channels[0][7]
        density = 1.5 * first @ first.conj().T - 0.5 * second @ second.conj().T
        rebuilt = (phi * natural) @ phi.conj().T
        # the n_occ largest natural occupations carry the density matrix to second order in the step
        self.assertLess(float(np.max(np.abs(rebuilt - density))), 5.0e-3)
        np.testing.assert_allclose(phi.conj().T @ phi, np.eye(count), atol=1.0e-12)
        self.assertAlmostEqual(float(natural.sum()), float(count), places=10)
        # a time-reversal pair counts twice in the norm (the mesh share of its block)
        paired = OuterDIIS(enabled=True, tolerance=1.0e-6)
        for step, block in ((1, first), (2, second)):
            paired.append(OutputSet(step, [{7: BlochOccupiedBlock(block, occupations, 2.0)}], {step - 1: 1.0}))
        self.assertAlmostEqual(paired.traces[(1, 2)], 2.0 * diis.traces[(1, 2)], places=12)


class KPointSamplingTests(unittest.TestCase):
    def test_time_reversal_reduction_of_a_uniform_mesh(self) -> None:
        lattice = 5.0 * np.eye(3)
        b = 2.0 * np.pi / 5.0
        fractions = np.array([[i / 3.0, j / 2.0, 0.0] for i in range(3) for j in range(2)])
        k_points, weights = time_reversal_reduced(fractions * b, np.full(6, 1.0 / 6.0), lattice)
        # (0,0) and (0,1/2) are their own partners; (1/3, x) pairs with (2/3, x)
        self.assertEqual(len(k_points), 4)
        np.testing.assert_allclose(sorted(weights), [1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0])
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=14)

    def test_only_uniform_meshes_are_taken(self) -> None:
        """The sets of the review (equal weights, distinct counts multiplying to the size, not uniform) are refused."""

        from parsec_python.SCF.hybrid_periodic import uniform_mesh_counts

        def along_b1(values):
            return np.array([[value, 0.0, 0.0] for value in values])

        for values in ([0.0, 0.2, 0.8], [0.1, 0.9, 0.3, 0.7], [0.0, 0.5, 0.2, 0.8]):
            with self.assertRaises(NotImplementedError):
                uniform_mesh_counts(along_b1(values))
        self.assertEqual(uniform_mesh_counts(along_b1([0.0, 1.0 / 3.0, 2.0 / 3.0])), (3, 1, 1))
        self.assertEqual(uniform_mesh_counts(along_b1([-3 / 8, -1 / 8, 1 / 8, 3 / 8])), (4, 1, 1))
        grid = np.array([[i / 3.0, j / 2.0, 0.0] for i in range(3) for j in range(2)])
        self.assertEqual(uniform_mesh_counts(grid), (3, 2, 1))
        repeated = grid.copy()
        repeated[-1] = repeated[0]
        for points in (grid[:-1], repeated):
            with self.assertRaises(NotImplementedError):
                uniform_mesh_counts(points)

    def test_mesh_counts_of_a_written_out_list(self) -> None:
        from types import SimpleNamespace

        from parsec_python.SCF.hybrid_periodic import mesh_counts

        cell = PeriodicCell(lattice_vectors=np.array([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]]))
        reciprocal = 2.0 * np.pi * np.linalg.inv(cell.lattice_vectors).T
        fractions = np.array([[i / 2.0, j / 2.0, k / 2.0] for i in range(2) for j in range(2) for k in range(2)])
        problem = SimpleNamespace(
            monkhorst_pack_dimensions=None,
            kpoints=fractions @ reciprocal,
            kpoint_weights=np.full(8, 1.0 / 8.0),
            periodic_cell=cell,
        )
        self.assertEqual(mesh_counts(problem), (2, 2, 2))
        problem.kpoints = problem.kpoints[:5]
        problem.kpoint_weights = np.full(5, 0.2)
        with self.assertRaises(NotImplementedError):
            mesh_counts(problem)


def _h2_problem(functional="pbe0", spin=False, **extra):
    return SinglePointInput(
        atoms=[Atom("H", [3.0, 3.0, 2.3]), Atom("H", [3.0, 3.0, 3.7])],
        pseudopotentials={"H": SpeciesPotential(HYDROGEN, 0)},
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=6.0 * np.eye(3)),
        scf=SCFSettings(
            xc_functional=functional,
            exact_exchange=ExactExchangeSettings(tolerance=1.0e-9),
            max_iterations=80,
            number_of_states=3,
            convergence_criterion=1.0e-7,
            spin_polarized=spin,
        ),
        hartree=HartreeSettings(relative_tolerance=1e-10, absolute_tolerance=1e-13, max_iterations=4000),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-9),
        **extra,
    )


class PeriodicFockLoopTests(unittest.TestCase):
    """H2 in a 6-bohr box (10**3 points): the Fock loop through both engines."""

    @classmethod
    def setUpClass(cls) -> None:
        from parsec_python.SCF import prepare_periodic_single_point, run_scf
        from parsec_python.SCF.hybrid_periodic import run_periodic_hybrid_scf

        cls.system = prepare_periodic_single_point(_h2_problem())
        cls.gamma_records = []
        cls.gamma = run_scf(cls.system, callback=cls.gamma_records.append)
        cls.bloch = run_periodic_hybrid_scf(cls.system, np.zeros((1, 3)), np.ones(1))

    def test_gamma_real_and_bloch_loops_agree(self) -> None:
        self.assertTrue(self.gamma.converged and self.bloch.converged)
        self.assertIsNotNone(self.gamma.exact_exchange)
        self.assertAlmostEqual(self.bloch.energies.total, self.gamma.energies.total, delta=1.0e-8)
        self.assertAlmostEqual(
            self.bloch.exact_exchange.exchange_energy, self.gamma.exact_exchange.exchange_energy, delta=1.0e-7
        )
        self.assertLess(self.bloch.exact_exchange.estimated_error, 1.0e-9)
        # the hybrid lies above PBE for H2 (a_x E_x replaces a quarter of the more negative PBE exchange here)
        records = [record for record in self.gamma_records if type(record).__name__ == "ExchangeLoopIteration"]
        self.assertGreaterEqual(len(records), 3)
        self.assertTrue(records[-1].converged)

    def test_forces_and_the_next_structure_of_an_ionic_loop(self) -> None:
        from parsec_python.Relaxation.backend import periodic_backend

        backend = periodic_backend(np.zeros((1, 3)), np.ones(1))
        result = backend.run_scf(self.system)
        self.assertIsNotNone(result.exact_exchange)
        self.assertAlmostEqual(result.energies.total, self.bloch.energies.total, delta=1.0e-8)
        forces = backend.compute_forces(self.system, result)
        # the Bloch orbitals at k = 0 and the real ones at the Gamma point give the same forces
        gamma = periodic_backend()
        reference = gamma.compute_forces(self.system, self.gamma)
        np.testing.assert_allclose(forces.total, reference.total, rtol=0.0, atol=1.0e-6)
        self.assertGreater(float(np.max(np.abs(reference.total))), 1.0e-3)
        from dataclasses import replace

        moved_atoms = list(self.system.atoms)
        moved_atoms[1] = replace(moved_atoms[1], position=np.asarray(moved_atoms[1].position) + np.array([0.0, 0.0, 0.05]))
        moved = backend.update_geometry(self.system, moved_atoms)
        start = backend.warm_start(self.system, result, moved)
        self.assertIsNotNone(start.exchange_orbitals)
        records = []
        again = backend.run_scf(moved, warm_start=start, callback=records.append)
        self.assertTrue(again.converged)
        outer = [record for record in records if type(record).__name__ == "ExchangeLoopIteration"]
        # the orbitals before the move start the Fock loop: no semilocal start
        self.assertIsNone(outer[0].start_functional)

    def test_hartree_fock_with_k_points_and_outer_diis(self) -> None:
        """HF converges slowly enough for the outer DIIS; with it and without it the loop reaches one state."""

        from parsec_python.SCF import prepare_periodic_single_point
        from parsec_python.SCF.hybrid_periodic import run_periodic_hybrid_scf

        b1 = 2.0 * np.pi / 6.0
        k_points, k_weights = np.array([[0.25 * b1, 0.0, 0.0]]), np.ones(1)
        results = {}
        for mixing in ("diis", "none"):
            problem = _h2_problem("hf")
            settings = replace(problem.scf.exact_exchange, outer_mixing=mixing, tolerance=1.0e-10)
            problem = replace(problem, scf=replace(problem.scf, exact_exchange=settings))
            records = []
            result = run_periodic_hybrid_scf(
                prepare_periodic_single_point(problem), k_points, k_weights, callback=records.append
            )
            self.assertTrue(result.converged)
            outer = [record for record in records if type(record).__name__ == "ExchangeLoopIteration"]
            results[mixing] = (result, outer)
        self.assertAlmostEqual(results["diis"][0].energies.total, results["none"][0].energies.total, delta=1.0e-7)
        self.assertFalse(any(record.extrapolated for record in results["none"][1]))
        # measured: one extrapolated step (dexx 7e-7 -> 8e-9 instead of 1.2e-7), 9 outer iterations instead of 11
        self.assertTrue(any(record.extrapolated for record in results["diis"][1]))
        self.assertLess(len(results["diis"][1]), len(results["none"][1]))

    def test_spin_channel_without_electrons(self) -> None:
        """One electron (an H atom, fully polarized): the down channel has no exchange operator; the k-point loop
        and the Gamma loop on real orbitals (run_spin_scf, occup.in) reach the same energy."""

        from parsec_python.SCF import prepare_periodic_single_point, run_scf
        from parsec_python.SCF.hybrid_periodic import run_periodic_hybrid_scf
        from parsec_python.SCF.pbc import with_initial_spin_densities

        problem = _h2_problem(spin=True)
        problem = replace(
            problem,
            atoms=(Atom("H", [3.0, 3.0, 3.0]),),
            pseudopotentials={"H": SpeciesPotential(HYDROGEN, 0, initial_spin_polarization=1.0)},
            scf=replace(problem.scf, fermi_temperature_kelvin=10.0),
        )
        records = []
        bloch = run_periodic_hybrid_scf(
            prepare_periodic_single_point(problem), np.zeros((1, 3)), np.ones(1), callback=records.append
        )
        self.assertTrue(bloch.converged)
        self.assertAlmostEqual(bloch.magnetic_moment, 1.0, places=6)
        outer = [record for record in records if type(record).__name__ == "ExchangeLoopIteration"]
        self.assertEqual(outer[0].operator_ranks[1], 0)
        fixed = replace(
            problem,
            scf=replace(
                problem.scf, fermi_temperature_kelvin=-1.0, fixed_occupations=(1.0, 0.0, 0.0), fixed_occupations_down=(0.0, 0.0, 0.0)
            ),
        )
        gamma = run_scf(with_initial_spin_densities(prepare_periodic_single_point(fixed)))
        self.assertTrue(gamma.converged)
        self.assertAlmostEqual(bloch.energies.total, gamma.energies.total, delta=2.0e-7)

    def test_two_equal_spin_channels_give_the_unpolarized_loop(self) -> None:
        from dataclasses import replace

        from parsec_python.SCF import prepare_periodic_single_point
        from parsec_python.SCF.hybrid_periodic import run_periodic_hybrid_scf

        system = prepare_periodic_single_point(_h2_problem(spin=True))
        result = run_periodic_hybrid_scf(system, np.zeros((1, 3)), np.ones(1))
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.energies.total, self.bloch.energies.total, delta=2.0e-7)
        self.assertAlmostEqual(result.magnetic_moment, 0.0, delta=1.0e-6)
        self.assertIsNotNone(replace(result).exact_exchange)


class PeriodicHybridInputTests(unittest.TestCase):
    BASE = """Boundary_Conditions: {boundary}
Periodic_System: .true.
begin Cell_Shape
{cell}end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.6
Expansion_Order: 6
States_Num: 3
Atom_Types_Num: 1
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
  3.0 3.0 2.3
  3.0 3.0 3.7
end Atom_Coord
Correlation_Type: pbe0
{extra}"""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="parsec_periodic_hybrid_"))
        shutil.copy(HYDROGEN, self.directory / "H_POTRE.DAT")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def translation(self, boundary="bulk", cell="6.0 0 0\n0 6.0 0\n0 0 6.0\n", extra=""):
        from parsec_python.Input import parse_parsec_input

        path = self.directory / "parsec.in"
        path.write_text(self.BASE.format(boundary=boundary, cell=cell, extra=extra))
        return parse_parsec_input(path)

    def test_bulk_hybrid_is_read_and_its_mesh_reduced_by_time_reversal(self) -> None:
        from parsec_python.Grid.kpoints import kpoint_sampling

        translation = self.translation(extra="Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n3 1 1\nend Monkhorst_Pack_Grid\n")
        self.assertTrue(translation.problem.scf.is_hybrid)
        k_points, weights = kpoint_sampling(translation.problem)
        self.assertEqual(len(k_points), 2)
        np.testing.assert_allclose(sorted(weights), [1.0 / 3.0, 2.0 / 3.0])

    def test_reference_command_line_runs_k_points_and_reports_band_edges(self) -> None:
        from parsec_python.cli import main

        self.translation(
            extra=(
                "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 1 1\nend Monkhorst_Pack_Grid\n"
                "Exx_Tolerance: 1e-7 Ry\nConvergence_Criterion: 1e-6 Ry\nExx_Divergence: gygi-baldereschi-x\n"
            )
        )
        code = main([str(self.directory / "parsec.in"), "--quiet", "--no-archive"])
        self.assertEqual(code, 0)
        text = (self.directory / "parsec.out").read_text()
        self.assertIn("Fock loop converged", text)
        self.assertIn("1 k-points of the time-reversal-reduced mesh 2x1x1", text)
        self.assertIn("x_gamma extrapolation: all components", text)
        self.assertIn("Band gap (direct)", text)
        self.assertIn("Converged eigenvalues by k-point", text)

    def test_exchange_keywords_of_periodic_cells_are_refused_for_clusters_and_bulk(self) -> None:
        from parsec_python.Input import UnsupportedParsecOptionError

        with self.assertRaises(UnsupportedParsecOptionError):
            self.translation(extra="Exx_Slab_Coulomb: periodic\n")
        translation = self.translation(extra="Exx_Divergence: none\nExx_Auxiliary_Exponent: 0.2\n")
        settings = translation.problem.scf.exact_exchange
        self.assertEqual((settings.divergence, settings.auxiliary_exponent), ("none", 0.2))
        self.assertEqual(repr(self.translation(extra="Exx_Divergence: auto\n").problem.scf.exact_exchange), "ExactExchangeSettings()")

    def test_wire_and_cg_pair_solver_are_refused(self) -> None:
        from parsec_python.Input import UnsupportedParsecOptionError

        with self.assertRaises(UnsupportedParsecOptionError):
            self.translation(extra="Exx_Pair_Solver: cg\n")
        with self.assertRaises(UnsupportedParsecOptionError):
            self.translation(boundary="wire", cell="6.0 0 0\n", extra="Boundary_Sphere_Radius: 3.0\n")


if __name__ == "__main__":
    unittest.main()
