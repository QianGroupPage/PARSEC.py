"""Component tests of the LDA+U term (``upot.F90``, ``calcden.f90z``, ``plusu.f90z``).

The projectors, the occupation and potential matrices, the energies and the
Hamiltonian term are compared with loops written after the Fortran, and the
forces with finite differences of the energy at frozen orbitals.  The
comparison of whole SCF runs with Fortran PARSEC is in the report of the
parity round.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python.Forces.hubbard_force import (
    build_hubbard_projector_gradients,
    contract_hubbard_forces,
    hubbard_forces,
)
from parsec_python.Grid import build_cluster_grid
from parsec_python.Hamiltonian import KohnShamHamiltonian
from parsec_python.Hamiltonian.hubbard import (
    HubbardOperator,
    build_hubbard_projectors,
    coulomb_tensors,
    hubbard_cutoff_radius,
    hubbard_terms,
    hubbard_terms_from_orbitals,
    interpolate_hubbard_radial,
    occupation_matrices,
)
from parsec_python.Hamiltonian.hubbard_tensors import P3_ENTRIES, P4_ENTRIES
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.V_ion import real_spherical_harmonics
from parsec_python.models import Atom, GridSettings, SCFSettings, SpeciesPotential


DATA = Path(__file__).resolve().parents[1] / "data"
AG = DATA / "Ag_POTRE.DAT"


def _fortran_source(name: str) -> Path | None:
    """The PARSEC source file ``name`` where a checkout has it, else ``None``."""

    candidates = []
    if os.environ.get("PARSEC_FORTRAN_SOURCE"):
        candidates.append(Path(os.environ["PARSEC_FORTRAN_SOURCE"]))
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / ".tmp" / "fortran_parity_20261027" / "src")
        candidates.append(parent / "src")
    for directory in candidates:
        if (directory / name).is_file():
            return directory / name
    return None


def _fortran15_source(relative: str) -> Path | None:
    """The PARSEC 1.5 source file ``relative`` (e.g. ``potential/upot.F90``), else ``None``."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        for directory in (
            parent / "f15" / "src",
            parent / ".tmp" / "parity15_20261104" / "f15" / "src",
        ):
            if (directory / relative).is_file():
                return directory / relative
    return None


def _fortran_rc(potential) -> float:
    """The loop of ``upot.F90`` with Fortran indices (rs(1) = 0)."""

    rs = np.concatenate(([0.0], potential.radii, [0.0]))
    wf = np.concatenate(([0.0], potential.radial_wavefunctions[2] / potential.radii, [0.0]))
    tmpnorm = (wf[0] * rs[0]) ** 2
    for i in range(1, potential.radii.size + 1):
        fact1 = 0.5 * (rs[i + 1] - rs[i - 1])
        tmpnorm += fact1 * (wf[i] * rs[i]) ** 2
        if tmpnorm > 0.95:
            return float(rs[i])
    raise AssertionError("no radius")


def _fortran_interpolation(potential, r: float) -> float:
    rs = np.concatenate(([0.0], potential.radii))
    wf = np.concatenate(([0.0], potential.radial_wavefunctions[2] / potential.radii))
    rnorm = max(r, rs[1])
    npt = int((1.0 / potential.radial_parameter_b) * np.log(1.0 + (1.0 / potential.radial_parameter_a) * rnorm)) + 1
    j = npt - 1
    fact1 = (rnorm - rs[j]) / (rs[j + 1] - rs[j])
    return float(wf[j] + fact1 * (wf[j + 1] - wf[j]))


def _ag_system(positions, *, u=0.4, j=0.0, radius=7.0, spacing=0.6):
    potential = read_parsec_pseudopotential(AG)
    specifications = {"Ag": SpeciesPotential(AG, 0, element_symbol="Ag", hubbard_u=u, hubbard_j=j)}
    grid = build_cluster_grid(GridSettings(radius=radius, spacing=spacing))
    atoms = tuple(Atom("Ag", position) for position in positions)
    projectors = build_hubbard_projectors(grid, atoms, {"Ag": potential}, specifications)
    return grid, atoms, {"Ag": potential}, specifications, projectors


def _loop_terms(occupations, u, j, p3, p4):
    """``calcden.f90z`` with its loops: D and the two energies."""

    nspin = len(occupations)
    atoms = occupations[0].shape[0]
    potentials = np.zeros((nspin, atoms, 5, 5))
    first = 0.0
    for spin in range(nspin):
        for a in range(atoms):
            for m in range(5):
                first += (u[a] - j[a]) / 4.0 * occupations[spin][a, m, m]
    for a in range(atoms):
        for m in range(5):
            for mp in range(5):
                for spin in range(nspin):
                    other = occupations[1 - spin] if nspin == 2 else occupations[spin]
                    same = sum(p3[m, mp, k, l] * occupations[spin][a, k, l] for k in range(5) for l in range(5))
                    cross = sum(p4[m, mp, k, l] * other[a, k, l] for k in range(5) for l in range(5))
                    value = -u[a] * occupations[spin][a, m, mp] + 2.0 * j[a] * (same + cross)
                    potentials[spin, a, m, mp] = value.real
            potentials[:, a, m, m] += 0.5 * u[a] - 0.5 * j[a]
    second = 0.0
    for spin in range(nspin):
        for a in range(atoms):
            second -= 0.5 * np.sum(potentials[spin, a] * occupations[spin][a])
    factor = 2.0 / nspin
    return potentials, factor * first, factor * (first + second)


def _random_occupations(rng, atoms, spins):
    result = []
    for _ in range(spins):
        matrices = []
        for _ in range(atoms):
            vectors = np.linalg.qr(rng.normal(size=(5, 5)))[0]
            weights = rng.uniform(0.0, 1.0, size=5)
            matrices.append(vectors @ np.diag(weights) @ vectors.T)
        result.append(np.array(matrices))
    return result


class HubbardTensorSourceTest(unittest.TestCase):
    # The previous tree writes ``cmplx(re, im)``, PARSEC 1.5 ``cmplx(re d0, im d0, dpc)``.
    PATTERN = re.compile(
        r"u_pot%p([34])\((\d),(\d),(\d),(\d)\)\s*=\s*cmplx\(\s*([-\d.]+)(?:d0)?,"
        r"\s*([-\d.]+)(?:d0)?\s*(,\s*dpc\s*)?\)"
    )

    def _rows(self, source: Path) -> tuple[dict[str, list], set[bool]]:
        text = source.read_text()
        body = text[text.index("subroutine initialize_p_matrices") :]
        rows = {"3": [], "4": []}
        kinds = set()
        for match in self.PATTERN.finditer(body):
            which, a, b, c, d, real, imaginary, kind = match.groups()
            rows[which].append((int(a), int(b), int(c), int(d), real, imaginary))
            kinds.add(kind is not None)
        return rows, kinds

    def test_tensor_rows_are_those_of_upot(self):
        source = _fortran_source("upot.F90")
        if source is None:
            self.skipTest("PARSEC source upot.F90 not available")
        rows, _ = self._rows(source)
        self.assertEqual(tuple(rows["3"]), P3_ENTRIES)
        self.assertEqual(tuple(rows["4"]), P4_ENTRIES)

    def test_tensor_rows_are_the_double_literals_of_parsec15(self):
        source = _fortran15_source("potential/upot.F90")
        if source is None:
            self.skipTest("PARSEC 1.5 source potential/upot.F90 not available")
        rows, kinds = self._rows(source)
        self.assertEqual(kinds, {True})
        self.assertEqual(tuple(rows["3"]), P3_ENTRIES)
        self.assertEqual(tuple(rows["4"]), P4_ENTRIES)

    def test_tensors_are_double_precision_literals(self):
        # PARSEC 1.5 (upot.F90:765-1522) stores the literals in double
        # precision; the previous tree rounded them to float32.
        p3, p4 = coulomb_tensors()
        self.assertEqual(p3.shape, (5, 5, 5, 5))
        self.assertEqual(np.count_nonzero(p3) <= len(P3_ENTRIES), True)
        m, mp, k, l, real, imaginary = P3_ENTRIES[1]
        value = p3[m - 1, mp - 1, k - 1, l - 1]
        self.assertEqual(value.real, float(real))
        self.assertEqual(value.imag, float(imaginary))
        self.assertNotEqual(value.real, float(np.float32(real)))
        self.assertFalse(p4.flags.writeable)


class HubbardProjectorTest(unittest.TestCase):
    def test_cutoff_radius_follows_the_fortran_loop(self):
        potential = read_parsec_pseudopotential(AG)
        self.assertEqual(hubbard_cutoff_radius(potential), _fortran_rc(potential))

    def test_interpolation_follows_the_fortran_index(self):
        potential = read_parsec_pseudopotential(AG)
        rng = np.random.default_rng(3)
        distances = np.concatenate(([0.0, 1.0e-9], rng.uniform(0.0, 2.5, size=200)))
        values, slopes = interpolate_hubbard_radial(potential, distances)
        expected = [_fortran_interpolation(potential, r) for r in distances]
        np.testing.assert_allclose(values, expected, rtol=1e-13, atol=1e-15)
        # The slope is that of the interpolant inside an interval.
        r = distances[5:]
        step = 1.0e-9
        upper, _ = interpolate_hubbard_radial(potential, r + step)
        lower, _ = interpolate_hubbard_radial(potential, r - step)
        np.testing.assert_allclose(slopes[5:], (upper - lower) / (2 * step), rtol=1e-4, atol=1e-6)
        self.assertEqual(slopes[0], 0.0)

    def test_projectors_are_sampled_radial_times_harmonic(self):
        position = np.array([0.13, -0.21, 0.07])
        grid, atoms, potentials, _, projectors = _ag_system([position])
        rc = projectors.cutoff_radii["Ag"]
        relative = grid.coordinates - position
        distance = np.linalg.norm(relative, axis=1)
        inside = np.flatnonzero(distance <= rc)
        self.assertEqual(projectors.support_sizes, (inside.size,))
        radial = np.array([_fortran_interpolation(potentials["Ag"], r) for r in distance[inside]])
        harmonics = real_spherical_harmonics(2, relative[inside])
        expected = np.sqrt(grid.volume_element) * radial[:, None] * harmonics
        dense = projectors.projectors.toarray()
        np.testing.assert_allclose(dense[inside], expected, rtol=1e-12, atol=1e-15)
        outside = np.setdiff1d(np.arange(grid.size), inside)
        self.assertEqual(np.count_nonzero(dense[outside]), 0)
        # Not renormalized: the cut keeps about 95 per cent of the norm.
        norms = np.sum(dense * dense, axis=0)
        self.assertTrue(np.all(norms < 1.0) and np.all(norms > 0.8))

    def test_periodic_images_add_into_the_same_rows(self):
        potential = read_parsec_pseudopotential(AG)
        specifications = {"Ag": SpeciesPotential(AG, 0, element_symbol="Ag", hubbard_u=0.3)}
        side = 4.2
        coordinates = np.stack(
            np.meshgrid(*(np.arange(0.0, side, 0.6),) * 3, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        grid = SimpleNamespace(coordinates=coordinates, volume_element=0.6**3, size=coordinates.shape[0])
        atoms = (Atom("Ag", [0.1, 0.2, 0.3]),)
        lattice = np.diag([side] * 3)
        projectors = build_hubbard_projectors(grid, atoms, {"Ag": potential}, specifications, lattice)
        rc = projectors.cutoff_radii["Ag"]
        expected = np.zeros((grid.size, 5))
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    relative = coordinates + np.array([i, j, k]) * side - atoms[0].position
                    distance = np.linalg.norm(relative, axis=1)
                    inside = distance <= rc
                    radial, _ = interpolate_hubbard_radial(potential, distance[inside])
                    expected[inside] += (
                        np.sqrt(grid.volume_element)
                        * radial[:, None]
                        * real_spherical_harmonics(2, relative[inside])
                    )
        np.testing.assert_allclose(projectors.projectors.toarray(), expected, atol=1e-14)
        with self.assertRaises(ValueError):
            build_hubbard_projectors(grid, atoms, {"Ag": potential}, specifications, np.diag([1.5] * 3))

    def test_no_projectors_without_parameters(self):
        potential = read_parsec_pseudopotential(AG)
        grid = build_cluster_grid(GridSettings(radius=4.0, spacing=0.8))
        specifications = {"Ag": SpeciesPotential(AG, 0, element_symbol="Ag")}
        self.assertIsNone(
            build_hubbard_projectors(grid, (Atom("Ag", [0, 0, 0]),), {"Ag": potential}, specifications)
        )


class HubbardAlgebraTest(unittest.TestCase):
    def setUp(self):
        self.p3, self.p4 = coulomb_tensors()
        self.projectors = SimpleNamespace(
            atom_count=2,
            hubbard_u=np.array([0.4, 0.25]),
            hubbard_j=np.array([0.07, 0.03]),
        )

    def test_terms_follow_calcden_loops(self):
        rng = np.random.default_rng(7)
        for spins in (1, 2):
            occupations = _random_occupations(rng, 2, spins)
            terms = hubbard_terms(self.projectors, occupations)
            potentials, first, energy = _loop_terms(
                occupations, self.projectors.hubbard_u, self.projectors.hubbard_j, self.p3, self.p4
            )
            np.testing.assert_allclose(terms.potentials, potentials, rtol=1e-13, atol=1e-15)
            self.assertAlmostEqual(terms.first_contribution, first, places=13)
            self.assertAlmostEqual(terms.energy, energy, places=13)

    def test_j_term_breaks_the_symmetry_of_d(self):
        rng = np.random.default_rng(11)
        terms = hubbard_terms(self.projectors, _random_occupations(rng, 2, 1))
        asymmetry = np.abs(terms.potentials - terms.potentials.transpose(0, 1, 3, 2)).max()
        self.assertGreater(asymmetry, 1.0e-3)

    def test_dudarev_limit(self):
        rng = np.random.default_rng(5)
        projectors = SimpleNamespace(
            atom_count=2, hubbard_u=np.array([0.4, 0.25]), hubbard_j=np.zeros(2)
        )
        for spins in (1, 2):
            occupations = _random_occupations(rng, 2, spins)
            terms = hubbard_terms(projectors, occupations)
            u = projectors.hubbard_u[None, :, None, None]
            np.testing.assert_allclose(
                terms.potentials,
                -u * np.asarray(occupations) + 0.5 * u * np.eye(5),
                atol=1e-15,
            )
            dudarev = 2.0 / spins * sum(
                0.5 * projectors.hubbard_u[a] * np.trace(n[a] - n[a] @ n[a])
                for n in occupations
                for a in range(2)
            )
            self.assertAlmostEqual(terms.band_energy + terms.energy, dudarev, places=13)
            # E_U is the derivative-consistent functional for J = 0.
            step = 1.0e-6
            direction = [rng.normal(size=(2, 5, 5)) for _ in range(spins)]
            direction = [d + d.transpose(0, 2, 1) for d in direction]

            def total(scale):
                shifted = [n + scale * d for n, d in zip(occupations, direction)]
                value = hubbard_terms(projectors, shifted)
                return value.band_energy + value.energy

            numeric = (total(step) - total(-step)) / (2 * step)
            analytic = 2.0 / spins * sum(
                np.sum(terms.potentials[s] * direction[s]) for s in range(spins)
            )
            self.assertAlmostEqual(numeric, analytic, places=7)

    def test_occupation_matrix_index_order(self):
        rng = np.random.default_rng(1)
        matrix = np.zeros((40, 10))
        matrix[:20, :5] = rng.normal(size=(20, 5))
        matrix[20:, 5:] = rng.normal(size=(20, 5))
        import scipy.sparse as sp

        projectors = SimpleNamespace(atom_count=2, projectors=sp.csc_matrix(matrix))
        projectors.overlaps = lambda vectors: (matrix.T @ vectors).reshape(2, 5, vectors.shape[1])
        orbitals = rng.normal(size=(40, 4))
        weights = np.array([1.0, 0.5, 0.0, 0.25])
        result = occupation_matrices(projectors, orbitals, weights)
        dots = (matrix.T @ orbitals).reshape(2, 5, 4)
        expected = np.zeros((2, 5, 5))
        for a in range(2):
            for im in range(5):
                for imp in range(5):
                    expected[a, imp, im] = np.sum(weights * dots[a, im] * dots[a, imp])
        np.testing.assert_allclose(result, expected, rtol=1e-13)


class HubbardOperatorTest(unittest.TestCase):
    def test_operator_matches_plusu_and_is_symmetric_for_j_zero(self):
        grid, atoms, potentials, _, projectors = _ag_system(
            [[0.0, 0.0, -1.3], [0.2, 0.1, 1.4]], radius=5.0, spacing=0.7
        )
        rng = np.random.default_rng(2)
        terms = hubbard_terms(projectors, _random_occupations(rng, 2, 1))
        operator = HubbardOperator(projectors, terms.potentials[0])
        vectors = rng.normal(size=(grid.size, 3))
        dense = projectors.projectors.toarray()
        expected = np.zeros_like(vectors)
        for a in range(2):
            block = dense[:, 5 * a : 5 * a + 5]
            dots = block.T @ vectors
            expected += block @ (terms.potentials[0, a] @ dots)
        np.testing.assert_allclose(operator.apply(vectors), expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(operator.apply(vectors[:, 0]), expected[:, 0], rtol=1e-12, atol=1e-14)
        matrix = operator.as_sparse().toarray()
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-14)
        np.testing.assert_allclose(matrix @ vectors, expected, rtol=1e-12, atol=1e-13)

    def test_kohn_sham_hamiltonian_adds_the_term(self):
        import scipy.sparse as sp

        grid, atoms, potentials, _, projectors = _ag_system([[0.0, 0.0, 0.0]], radius=4.0, spacing=0.8)
        rng = np.random.default_rng(4)
        terms = hubbard_terms(projectors, _random_occupations(rng, 1, 1))
        operator = HubbardOperator(projectors, terms.potentials[0])
        laplacian = sp.identity(grid.size, format="csr")
        nonlocal_operator = SimpleNamespace(
            shape=(grid.size, grid.size), apply=lambda v: np.zeros_like(v)
        )
        plain = KohnShamHamiltonian(laplacian, np.zeros(grid.size), nonlocal_operator)
        with_u = KohnShamHamiltonian(laplacian, np.zeros(grid.size), nonlocal_operator, operator)
        vectors = rng.normal(size=(grid.size, 2))
        np.testing.assert_allclose(with_u.apply(vectors) - plain.apply(vectors), operator.apply(vectors), atol=1e-14)


class HubbardForceTest(unittest.TestCase):
    def test_forces_are_the_derivative_at_frozen_orbitals(self):
        positions = np.array([[0.05, -0.1, -1.25], [0.15, 0.08, 1.35]])
        grid, atoms, potentials, specifications, projectors = _ag_system(positions, u=0.5)
        rng = np.random.default_rng(9)
        orbitals = np.linalg.qr(rng.normal(size=(grid.size, 6)))[0]
        occupations = np.array([1.0, 1.0, 0.8, 0.5, 0.2, 0.0])
        system = SimpleNamespace(
            hubbard=projectors,
            atoms=atoms,
            grid=grid,
            pseudopotentials=potentials,
            input=SimpleNamespace(periodic_cell=None),
        )
        forces = hubbard_forces(system, [orbitals], [occupations])

        def energy(moved):
            moved_atoms = tuple(Atom("Ag", position) for position in moved)
            moved_projectors = build_hubbard_projectors(grid, moved_atoms, potentials, specifications)
            self.assertEqual(moved_projectors.support_sizes, projectors.support_sizes)
            terms = hubbard_terms_from_orbitals(moved_projectors, [orbitals], [occupations])
            return terms.band_energy + terms.energy

        step = 1.0e-5
        for atom in range(2):
            for component in range(3):
                plus = positions.copy()
                minus = positions.copy()
                plus[atom, component] += step
                minus[atom, component] -= step
                numeric = -(energy(plus) - energy(minus)) / (2 * step)
                self.assertAlmostEqual(forces[atom, component], numeric, delta=2e-6 + 1e-4 * abs(numeric))

    def test_forces_refused_with_j(self):
        grid, atoms, potentials, _, projectors = _ag_system([[0, 0, 0]], j=0.05, radius=4.0, spacing=0.8)
        system = SimpleNamespace(
            hubbard=projectors, atoms=atoms, grid=grid, pseudopotentials=potentials,
            input=SimpleNamespace(periodic_cell=None),
        )
        with self.assertRaises(NotImplementedError):
            hubbard_forces(system, [np.zeros((grid.size, 1))], [np.ones(1)])


class HubbardSCFTest(unittest.TestCase):
    """A few iterations of the reference loops with the term."""

    def problem(self, spin: bool, u: float = 0.3):
        from parsec_python.models import (
            EigensolverSettings,
            HartreeSettings,
            SinglePointInput,
        )

        return SinglePointInput(
            atoms=[Atom("Ag", [0.0, 0.0, 0.0])],
            pseudopotentials={
                "Ag": SpeciesPotential(AG, 0, element_symbol="Ag", hubbard_u=u, initial_spin_polarization=0.3)
            },
            grid=GridSettings(spacing=0.7, radius=5.0, expansion_order=6),
            scf=SCFSettings(
                max_iterations=4,
                number_of_states=9,
                spin_polarized=spin,
                convergence_criterion=1.0e-9,
                fermi_temperature_kelvin=1000.0,
            ),
            hartree=HartreeSettings(multipole_order=4),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
        )

    def test_terms_enter_from_the_second_iteration(self):
        from parsec_python.Energy import spin_total_energy, total_energy
        from parsec_python.SCF.single_point import prepare_single_point, run_scf
        from parsec_python.models import CorrectedEnergyBreakdown, EnergyBreakdown

        for spin in (False, True):
            system = prepare_single_point(self.problem(spin))
            self.assertIsNotNone(system.hubbard)
            result = run_scf(system)
            first, *later = result.history
            self.assertIsNone(first.hubbard_energy)
            self.assertIsInstance(first.energies, CorrectedEnergyBreakdown)
            self.assertEqual(first.energies.hubbard, 0.0)
            for step in later:
                self.assertIsNotNone(step.hubbard_first_contribution)
                self.assertEqual(step.energies.hubbard, step.hubbard_energy)
            self.assertIsNotNone(result.hubbard)
            self.assertEqual(result.hubbard.spin_count, 2 if spin else 1)
            self.assertEqual(result.energies.hubbard, result.hubbard.energy)
            # The electronic energy is that of totnrg.f90 plus etot_plusu.
            if not spin:
                plain = total_energy(
                    result.eigenvalues,
                    result.occupations,
                    result.density,
                    result.input_effective_potential,
                    result.ionic_potential,
                    result.hartree_potential,
                    result.xc_potential,
                    result.energies.exchange_correlation,
                    result.energies.ion_ion,
                    system.grid.volume_element,
                )
                self.assertIs(type(plain), EnergyBreakdown)
                self.assertAlmostEqual(
                    result.energies.total, plain.total + result.hubbard.energy, places=10
                )

    def test_unchanged_without_parameters(self):
        from parsec_python.SCF.single_point import prepare_single_point, run_scf
        from parsec_python.models import EnergyBreakdown

        system = prepare_single_point(self.problem(False, u=0.0))
        self.assertIsNone(system.hubbard)
        result = run_scf(system)
        self.assertIs(type(result.energies), EnergyBreakdown)
        self.assertIsNone(result.history[-1].hubbard_energy)
        self.assertIsNone(result.hubbard)


class HubbardSettingsTest(unittest.TestCase):
    def test_repr_hides_zero_parameters(self):
        plain = SpeciesPotential(AG, 0)
        self.assertNotIn("hubbard_", repr(plain))
        self.assertIn("hubbard_u=0.3", repr(SpeciesPotential(AG, 0, hubbard_u=0.3)))
        self.assertNotIn("van_der_waals", repr(SCFSettings()))
        self.assertNotIn("hirshfeld", repr(SCFSettings()))
        self.assertIn("hirshfeld_cells=(1, 0, 2)", repr(SCFSettings(van_der_waals=True, hirshfeld_cells=(1, 0, 2))))


_INPUT = """
Boundary_Conditions: cluster
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.8
Coordinate_Unit: Cartesian_Bohr
States_Num: 10
Correlation_Type: {correlation}
Ignore_Symmetry: {ignore}
{extra}
Atom_Types_Num: 2
Atom_Type: Ag
Local_Component: s
{ag_line}
begin Atom_Coord
  0.0 0.0 -1.0
end Atom_Coord
Atom_Type: H
Local_Component: s
{h_line}
begin Atom_Coord
  0.0 0.0 1.0
end Atom_Coord
"""


class HubbardInputTest(unittest.TestCase):
    def parse(self, **fields):
        from parsec_python.Input import parse_parsec_input

        values = dict(correlation="ca", ignore=".true.", extra="", ag_line="ldaplusu_u 5.0 eV", h_line="")
        values.update(fields)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parsec.in"
            path.write_text(_INPUT.format(**values))
            for name, source in (("Ag_POTRE.DAT", AG), ("H_POTRE.DAT", DATA / "water_H_POTRE.DAT")):
                (Path(directory) / name).write_bytes(source.read_bytes())
            return parse_parsec_input(path)

    def test_parameters_in_ry_with_the_esdf_electronvolt(self):
        translation = self.parse(ag_line="ldaplusu_u 5.0 eV\nldaplusu_j 0.5 eV")
        specification = translation.problem.pseudopotentials["Ag"]
        self.assertAlmostEqual(specification.hubbard_u, 5.0 * 1.60219e-19 / 2.17991e-18, places=15)
        self.assertAlmostEqual(specification.hubbard_j, 0.5 * 1.60219e-19 / 2.17991e-18, places=15)
        self.assertFalse(translation.problem.pseudopotentials["H"].has_hubbard)
        self.assertTrue(any("ldaplusu_j" in warning for warning in translation.warnings))
        self.assertAlmostEqual(
            self.parse(ag_line="ldaplusu_u 0.3").problem.pseudopotentials["Ag"].hubbard_u, 0.3
        )

    def test_a_line_in_the_section_of_another_species_is_refused(self):
        from parsec_python.Input import ParsecInputError

        with self.assertRaises(ParsecInputError):
            self.parse(ag_line="", h_line="ldaplusu_u 0.3")

    def test_species_without_d_channel_is_ignored_with_a_warning(self):
        translation = self.parse(ag_line="ldaplusu_u 0", h_line="ldaplusu_u 0.3")
        self.assertFalse(translation.problem.pseudopotentials["H"].has_hubbard)
        self.assertTrue(any("no d channel" in warning for warning in translation.warnings))

    def test_refused_combinations(self):
        from parsec_python.Input import UnsupportedParsecOptionError

        with self.assertRaises(UnsupportedParsecOptionError):
            self.parse(ag_line="ldaplusu_u 0.3\nldaplusu_j 0.02", extra="Skip_Force: .false.")
        # The refinement replaces the XC energy of a total that holds E_U.
        self.assertEqual(
            self.parse(extra="Energy_Refinement: b3lyp").energy_refinement, "b3lyp"
        )
        translation = self.parse(extra="Skip_Force: .false.")
        self.assertTrue(translation.compute_forces)

    def test_accelerated_cli_routes_only_what_the_devices_lack(self):
        from parsec_python.acceleration.cli import _reference_only_features

        # J = 0 on a cluster runs on the accelerated driver
        # (acceleration.Hamiltonian.hubbard_device); a J term stays here.
        self.assertEqual(
            _reference_only_features(self.parse(), accelerated_backend_selected=False), []
        )
        features = _reference_only_features(
            self.parse(ag_line="ldaplusu_u 5.0 eV\nldaplusu_j 0.5 eV"),
            accelerated_backend_selected=False,
        )
        self.assertIn("LDA+U with ldaplusu_j != 0", features)
        plain = self.parse(ag_line="")
        self.assertEqual(_reference_only_features(plain, accelerated_backend_selected=False), [])


if __name__ == "__main__":
    unittest.main()
