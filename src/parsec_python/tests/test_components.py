from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp

from parsec_python import (
    AndersonMixer,
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    KohnShamHamiltonian,
    MixingSettings,
    NonlocalProjectorOperator,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
    build_cluster_grid,
    build_negative_laplacian,
    ca_lda,
    density_from_orbitals,
    density_multipoles,
    fermi_occupations,
    parsec_radial_integral,
    potential_residual_metrics,
    prepare_reference_single_point as prepare_single_point,
    read_parsec_pseudopotential,
    run_reference_scf as run_scf,
    second_derivative_coefficients,
    solve_hartree,
    total_energy,
)
from parsec_python.Eigensolvers import (
    ChebFFSettings,
    EigvalSettings,
    SubspaceSettings,
    solve_eigval,
)


class GridAndFiniteDifferenceTests(unittest.TestCase):
    def test_parsec_h2o_grid_counts(self) -> None:
        # Both ESDF inputs are exact multiples in angstrom, but the compiled
        # reference converts them independently and lands just below R/h=35.
        spacing = 0.2 * 1.8897268
        radius = np.nextafter(35.0 * spacing, -np.inf)
        half_shift = build_cluster_grid(
            GridSettings(
                spacing=spacing,
                radius=radius,
                expansion_order=8,
            )
        )
        zero_shift = build_cluster_grid(
            GridSettings(
                spacing=spacing,
                radius=radius,
                expansion_order=8,
                shift=(0.0, 0.0, 0.0),
            )
        )
        self.assertEqual(half_shift.size, 179944)
        self.assertEqual(zero_shift.size, 179309)

    def test_order_eight_coefficients_and_symmetric_operator(self) -> None:
        expected = np.array(
            [
                -1.0 / 560.0,
                8.0 / 315.0,
                -1.0 / 5.0,
                8.0 / 5.0,
                -205.0 / 72.0,
                8.0 / 5.0,
                -1.0 / 5.0,
                8.0 / 315.0,
                -1.0 / 560.0,
            ]
        )
        np.testing.assert_allclose(second_derivative_coefficients(8), expected)
        grid = build_cluster_grid(
            GridSettings(spacing=0.8, radius=2.4, expansion_order=8)
        )
        operator = build_negative_laplacian(grid)
        difference = operator - operator.T
        self.assertEqual(difference.nnz, 0)
        self.assertTrue(np.all(operator.diagonal() > 0))


class PhysicsKernelTests(unittest.TestCase):
    def test_ca_potential_is_energy_density_derivative(self) -> None:
        density = np.array([1.0e-4, 0.01, 0.2, 1.0])
        delta = density * 1.0e-6
        result = ca_lda(density, 1.0)
        plus = ca_lda(density + delta, 1.0).energy_density
        minus = ca_lda(density - delta, 1.0).energy_density
        numerical = (plus - minus) / (2.0 * delta)
        np.testing.assert_allclose(result.potential, numerical, rtol=3.0e-7)

    def test_occupations_and_density_conserve_electrons(self) -> None:
        eigenvalues = np.array([-1.0, -0.4, 0.2, 0.8])
        occupations = fermi_occupations(eigenvalues, 3.0, 80.0)
        self.assertAlmostEqual(2.0 * occupations.occupations.sum(), 3.0, places=10)
        q, _ = np.linalg.qr(np.arange(32, dtype=float).reshape(8, 4) + np.eye(8, 4))
        density = density_from_orbitals(q, occupations.occupations, 0.125)
        self.assertAlmostEqual(0.125 * density.sum(), 3.0, places=10)

    def test_zero_temperature_shares_a_degenerate_frontier(self) -> None:
        result = fermi_occupations(
            np.array([-1.0, 0.0, 0.0, 1.0]),
            electron_count=3.0,
            temperature_kelvin=0.0,
        )
        np.testing.assert_allclose(result.occupations, [1.0, 0.25, 0.25, 0.0])

    def test_monopole_boundary_has_rydberg_factor_two(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=1.0, radius=2.0, expansion_order=2)
        )
        density = np.ones(grid.size)
        expansion = density_multipoles(density, grid, order=0)
        electron_count = grid.integrate(density)
        points = np.array([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        np.testing.assert_allclose(
            expansion.potential(points),
            2.0 * electron_count / np.linalg.norm(points, axis=1),
        )

    def test_multipoles_match_direct_far_field(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=0.8, radius=2.0, expansion_order=4)
        )
        density = np.random.default_rng(4).random(grid.size)
        expansion = density_multipoles(density, grid, order=9)
        points = np.array([[20.0, 3.0, -1.0], [-4.0, 22.0, 2.0]])
        direct = np.array(
            [
                2.0
                * grid.volume_element
                * np.sum(density / np.linalg.norm(point - grid.coordinates, axis=1))
                for point in points
            ]
        )
        np.testing.assert_allclose(
            expansion.potential(points), direct, rtol=2.0e-10, atol=1.0e-11
        )

    def test_box_hartree_uses_exact_direct_boundary(self) -> None:
        grid = build_cluster_grid(
            GridSettings(
                spacing=1.0,
                radius=2.0,
                expansion_order=2,
                shift=(0.0, 0.0, 0.0),
                domain_shape="box",
                box_lengths=(4.0, 4.0, 4.0),
            )
        )
        density = np.zeros(grid.size)
        source_row = np.flatnonzero(
            np.all(grid.coordinates == np.array([2.0, 2.0, 2.0]), axis=1)
        )[0]
        density[source_row] = 1.0 / grid.volume_element
        result = solve_hartree(
            density,
            grid,
            build_negative_laplacian(grid),
            HartreeSettings(boundary_method="auto"),
        )
        self.assertTrue(result.converged)
        self.assertLess(result.residual_norm, 1.0e-7 * result.initial_residual_norm + 1.0e-13)
        self.assertAlmostEqual(
            result.boundary.potential(np.array([[3.0, 0.0, 0.0]]))[0],
            2.0 / 3.0,
            places=13,
        )

    def test_residual_and_first_anderson_step(self) -> None:
        input_potential = np.array([0.0, 1.0, 2.0])
        output_potential = np.array([1.0, 1.5, 1.0])
        density = np.array([0.5, 1.0, 0.5])
        metrics = potential_residual_metrics(
            input_potential, output_potential, density, 0.25, 0.5
        )
        expected = output_potential - input_potential
        np.testing.assert_allclose(metrics.residual, expected)
        self.assertAlmostEqual(metrics.plain, 0.75)
        self.assertAlmostEqual(metrics.weighted, np.sqrt(0.625))
        mixer = AndersonMixer(MixingSettings(parameter=0.3))
        np.testing.assert_allclose(
            mixer.mix(input_potential, output_potential, iteration=1),
            input_potential + 0.3 * expected,
        )
        second_input = input_potential + 0.3 * expected
        second_output = np.array([0.7, 1.1, 1.8])
        first_residual = output_potential - input_potential
        second_residual = second_output - second_input
        difference = second_residual - first_residual
        coefficient = np.dot(difference, second_residual) / np.dot(
            difference, difference
        )
        expected_second = (
            second_input
            + coefficient * (input_potential - second_input)
            + 0.3
            * (
                second_residual
                + coefficient * (first_residual - second_residual)
            )
        )
        np.testing.assert_allclose(
            mixer.mix(second_input, second_output, iteration=2),
            expected_second,
        )

    def test_anderson_safeguard_restarts_after_residual_growth(self) -> None:
        settings = MixingSettings(
            parameter=0.3,
            safeguard=True,
            growth_trigger=2.0,
            backoff=0.5,
        )
        mixer = AndersonMixer(settings)
        first_input = np.zeros(2)
        first_mixed = mixer.mix(first_input, np.array([1.0, 0.0]), iteration=1)
        np.testing.assert_allclose(first_mixed, np.array([0.3, 0.0]))

        second_output = first_mixed + np.array([3.0, 0.0])
        second_mixed = mixer.mix(first_mixed, second_output, iteration=2)
        np.testing.assert_allclose(second_mixed, first_mixed + np.array([0.45, 0.0]))
        self.assertEqual(mixer.safeguard_resets, 1)

    def test_anderson_safeguard_limits_extrapolated_step(self) -> None:
        settings = MixingSettings(
            parameter=0.3,
            safeguard=True,
            step_limit=2.0,
            growth_trigger=10.0,
        )
        mixer = AndersonMixer(settings)
        current = mixer.mix(np.zeros(2), np.array([1.0, 0.0]), iteration=1)
        residual = np.array([0.9, 0.001])
        mixed = mixer.mix(current, current + residual, iteration=2)
        self.assertLessEqual(
            np.linalg.norm(mixed - current),
            settings.step_limit * settings.parameter * np.linalg.norm(residual)
            + 1.0e-14,
        )
        self.assertEqual(mixer.safeguard_clips, 1)

    def test_chebyshev_reuse_branch_on_diagonal_operator(self) -> None:
        size = 30
        kinetic = sp.diags(np.linspace(0.0, 10.0, size), format="csr")
        empty_nonlocal = NonlocalProjectorOperator(
            projectors=sp.csc_matrix((size, 0)),
            signs=np.empty(0),
            labels=(),
        )
        settings = EigvalSettings(
            safety_buffer=6,
            chebff=ChebFFSettings(
                polynomial_degree=12,
                filter_cycles=2,
                block_size=6,
                random_seed=7,
            ),
            subspace=SubspaceSettings(
                polynomial_degree=12,
                degree_delta=2,
                block_size=6,
                random_seed=7,
            ),
        )
        first_operator = KohnShamHamiltonian(
            kinetic, np.zeros(size), empty_nonlocal
        ).as_linear_operator()
        first = solve_eigval(
            first_operator,
            3,
            settings=settings,
        )
        second_operator = KohnShamHamiltonian(
            kinetic, np.linspace(0.01, -0.01, size), empty_nonlocal
        ).as_linear_operator()
        second = solve_eigval(
            second_operator,
            3,
            settings=settings,
            state=first.state,
        )
        self.assertEqual(first.solver_path, "chebff")
        self.assertEqual(second.solver_path, "subspace")
        self.assertIsNotNone(second.residual_norms)
        self.assertLess(float(np.max(second.residual_norms)), 1.0e-8)

    def test_total_energy_input_hxc_double_counting(self) -> None:
        energy = total_energy(
            eigenvalues=np.array([-1.0, 0.2]),
            occupations=np.array([1.0, 0.25]),
            density=np.array([1.0, 2.0]),
            input_effective_potential=np.array([-1.1, -0.6]),
            ionic_potential=np.array([-2.0, -1.0]),
            output_hartree_potential=np.array([1.0, 0.5]),
            output_xc_potential=np.array([-0.3, -0.2]),
            exchange_correlation_energy=-0.4,
            ion_ion_energy=0.7,
            volume_element=0.5,
        )
        self.assertAlmostEqual(energy.eigenvalue, -1.9)
        self.assertAlmostEqual(energy.hartree, 0.5)
        self.assertAlmostEqual(energy.electronic, -2.65)
        self.assertAlmostEqual(energy.total, -1.95)


class PseudopotentialTests(unittest.TestCase):
    def test_martins_new_reader_and_tail(self) -> None:
        path = Path(__file__).parent / "data" / "H_POTRE.DAT"
        potential = read_parsec_pseudopotential(path)
        self.assertEqual(potential.symbol, "H")
        self.assertEqual(potential.radii.size, 6)
        self.assertEqual(sorted(potential.channel_potentials), [0, 1])
        self.assertFalse(potential.has_nonlinear_core_correction)
        self.assertAlmostEqual(
            potential.local_potential(np.array([10.0]), 0)[0], -0.2
        )
        cutoff = potential.interpolation_cutoff
        self.assertAlmostEqual(
            potential.local_potential(np.array([cutoff]), 0)[0],
            -2.0 * potential.ionic_charge / cutoff,
        )
        self.assertEqual(
            potential.interpolate_density(np.array([cutoff]))[0], 0.0
        )
        projector, sign = potential.radial_projector(1, 0)
        self.assertEqual(projector.shape, potential.radii.shape)
        self.assertIn(sign, (-1.0, 1.0))
        delta_v = (
            potential.channel_potentials[1]
            - potential.channel_potentials[0]
        )
        wave = potential.radial_wavefunctions[1]
        denominator = parsec_radial_integral(
            potential.radii,
            wave * wave * delta_v,
        )
        expected = delta_v * wave / potential.radii / np.sqrt(abs(denominator))
        np.testing.assert_allclose(projector, expected)

    def test_unhyphenated_benzene_wavefunction_marker(self) -> None:
        path = (
            Path(__file__).resolve().parents[3]
            / "examples"
            / "0d_benzene"
            / "C_POTRE.DAT"
        )
        potential = read_parsec_pseudopotential(path)
        self.assertEqual(potential.ionic_charge, 4.0)
        self.assertEqual(sorted(potential.channel_potentials), [0, 1])
        self.assertEqual(sorted(potential.radial_wavefunctions), [0, 1])
        _, sign = potential.radial_projector(0, 1)
        self.assertEqual(sign, 1.0)


class IntegrationTests(unittest.TestCase):
    def test_one_iteration_uses_only_modular_parsec_components(self) -> None:
        path = Path(__file__).parent / "data" / "H_POTRE.DAT"
        problem = SinglePointInput(
            atoms=[Atom("H", [0.0, 0.0, 0.0])],
            pseudopotentials={"H": SpeciesPotential(path, 0)},
            grid=GridSettings(spacing=0.8, radius=4.0, expansion_order=4),
            scf=SCFSettings(max_iterations=1, number_of_states=3),
            hartree=HartreeSettings(multipole_order=2),
            eigensolver=EigensolverSettings(
                method="chebff",
                tolerance=1.0e-6,
            ),
        )
        system = prepare_single_point(problem)
        for name, value in vars(system.timings).items():
            with self.subTest(preparation_timing=name):
                self.assertGreaterEqual(value, 0.0)
        probe = np.linspace(-1.0, 1.0, system.grid.size)
        np.testing.assert_allclose(
            system.nonlocal_operator.apply(probe),
            system.nonlocal_operator.as_sparse() @ probe,
            atol=1.0e-13,
        )
        result = run_scf(system)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(len(result.history[0].eigenvalues), 3)
        self.assertEqual(len(result.history[0].occupations), 3)
        self.assertTrue(np.isfinite(result.history[0].fermi_level))
        self.assertAlmostEqual(
            result.history[0].density_minimum,
            float(np.min(result.density)),
        )
        self.assertAlmostEqual(
            result.history[0].density_maximum,
            float(np.max(result.density)),
        )
        self.assertGreaterEqual(result.history[0].diagonalization_seconds, 0.0)
        self.assertGreaterEqual(result.history[0].hartree_seconds, 0.0)
        for name in (
            "hamiltonian_binding_seconds",
            "occupations_density_seconds",
            "xc_seconds",
            "mixing_energy_seconds",
            "total_seconds",
        ):
            with self.subTest(iteration_timing=name):
                self.assertGreaterEqual(getattr(result.history[0], name), 0.0)
        self.assertIs(result.timings.preparation, system.timings)
        for name, value in vars(result.timings).items():
            if name == "preparation":
                continue
            with self.subTest(run_timing=name):
                self.assertGreaterEqual(value, 0.0)
        self.assertAlmostEqual(
            system.grid.integrate(result.density), result.electron_count, places=9
        )
        self.assertTrue(np.isfinite(result.energies.total))
        applied = system.hamiltonian(result.input_effective_potential).apply(
            result.wavefunctions
        )
        residual = applied - result.wavefunctions * result.eigenvalues[None, :]
        # CHEBFF deliberately reports approximate convergence after its
        # fixed filter cycles; it does not impose a Ritz-residual threshold.
        self.assertLess(float(np.max(np.linalg.norm(residual, axis=0))), 1.0e-3)
        self.assertTrue(np.isnan(result.history[0].eigen_residual_max))
        np.testing.assert_allclose(
            result.output_effective_potential,
            result.ionic_potential
            + result.hartree_potential
            + result.xc_potential,
        )

    def test_orbital_operators_only_builds_the_same_hamiltonian_terms(self) -> None:
        path = Path(__file__).parent / "data" / "H_CORE_POTRE.DAT"
        problem = SinglePointInput(
            atoms=[Atom("H", [0.3, 0.0, -0.2]), Atom("H", [-0.9, 0.4, 0.5])],
            pseudopotentials={"H": SpeciesPotential(path, 0)},
            grid=GridSettings(spacing=0.8, radius=4.0, expansion_order=4),
            scf=SCFSettings(max_iterations=1, number_of_states=3),
            eigensolver=EigensolverSettings(method="chebff"),
        )
        complete = prepare_single_point(problem)

        def unused(*_args, **_kwargs):
            raise AssertionError("a field of the SCF loop was built")

        # The public driver takes the problem alone; this option belongs to
        # the builder-level entry point the accelerated package calls.
        from parsec_python.SCF import prepare_single_point as prepare_components

        reduced = prepare_components(
            problem,
            orbital_operators_only=True,
            local_ionic_builder=unused,
            atomic_density_builder=unused,
        )
        # Everything that enters H is the complete preparation, bit for bit.
        self.assertEqual(len(reduced.atoms), len(complete.atoms))
        for atom, expected in zip(reduced.atoms, complete.atoms, strict=True):
            self.assertEqual(atom.symbol, expected.symbol)
            np.testing.assert_array_equal(atom.position, expected.position)
        self.assertEqual(reduced.electron_count, complete.electron_count)
        np.testing.assert_array_equal(
            reduced.grid.coordinates, complete.grid.coordinates
        )
        for name in ("indptr", "indices", "data"):
            with self.subTest(buffer=name):
                np.testing.assert_array_equal(
                    getattr(reduced.negative_laplacian, name),
                    getattr(complete.negative_laplacian, name),
                )
                np.testing.assert_array_equal(
                    getattr(reduced.nonlocal_operator.projectors, name),
                    getattr(complete.nonlocal_operator.projectors, name),
                )
        np.testing.assert_array_equal(
            reduced.nonlocal_operator.signs, complete.nonlocal_operator.signs
        )
        self.assertGreater(reduced.nonlocal_operator.projectors.nnz, 0)
        # The fields of the SCF loop are absent, not empty or zero.
        self.assertIsNone(reduced.ionic_potential)
        self.assertIsNone(reduced.initial_density)
        self.assertIsNone(reduced.core_density)
        self.assertTrue(np.isnan(reduced.ion_ion_energy))
        self.assertTrue(np.isnan(reduced.atomic_reference_correction))
        self.assertTrue(np.isfinite(complete.ion_ion_energy))
        self.assertGreater(complete.ion_ion_energy, 0.0)
        for name in (
            "local_ionic_seconds",
            "initial_density_seconds",
            "core_density_seconds",
            "ion_ion_seconds",
        ):
            with self.subTest(preparation_timing=name):
                self.assertEqual(getattr(reduced.timings, name), 0.0)

    def test_completion_adds_the_fields_of_a_complete_preparation(self) -> None:
        from threading import Thread, get_ident

        from parsec_python.SCF import complete_single_point
        from parsec_python.SCF import prepare_single_point as prepare_components
        from parsec_python.V_ion import (
            build_local_ionic_potential,
            superpose_atomic_density,
        )

        path = Path(__file__).parent / "data" / "H_CORE_POTRE.DAT"
        problem = SinglePointInput(
            atoms=[Atom("H", [0.3, 0.0, -0.2]), Atom("H", [-0.9, 0.4, 0.5])],
            pseudopotentials={"H": SpeciesPotential(path, 0)},
            grid=GridSettings(spacing=0.8, radius=4.0, expansion_order=4),
            scf=SCFSettings(max_iterations=1, number_of_states=3),
            eigensolver=EigensolverSettings(method="chebff"),
        )
        complete = prepare_single_point(problem)
        reduced = prepare_components(problem, orbital_operators_only=True)

        # On another thread, where the accelerated driver runs it.
        outcome = []
        worker = Thread(target=lambda: outcome.append(complete_single_point(reduced)))
        worker.start()
        worker.join()
        (completed,) = outcome

        def assert_same_fields(actual) -> None:
            for name in ("ionic_potential", "initial_density", "core_density"):
                with self.subTest(field=name):
                    left, right = getattr(actual, name), getattr(complete, name)
                    self.assertEqual(left.dtype, right.dtype)
                    self.assertEqual(left.shape, right.shape)
                    # Bit for bit, the sign of a zero included.
                    self.assertEqual(left.tobytes(), right.tobytes())
            self.assertEqual(actual.ion_ion_energy, complete.ion_ion_energy)
            self.assertEqual(
                actual.atomic_reference_correction,
                complete.atomic_reference_correction,
            )

        assert_same_fields(completed)
        self.assertGreater(float(np.max(completed.core_density)), 0.0)
        # What enters H is shared with the reduced system, not built again.
        for name in (
            "input",
            "atoms",
            "pseudopotentials",
            "grid",
            "negative_laplacian",
            "nonlocal_operator",
        ):
            with self.subTest(shared=name):
                self.assertIs(getattr(completed, name), getattr(reduced, name))
        self.assertEqual(completed.electron_count, reduced.electron_count)
        for name in (
            "pseudopotential_loading_seconds",
            "grid_seconds",
            "finite_difference_seconds",
            "nonlocal_ionic_seconds",
            # The wall time of the call that left the fields out: the
            # completion may have run beside other work and adds nothing.
            "total_seconds",
        ):
            with self.subTest(kept_timing=name):
                self.assertEqual(
                    getattr(completed.timings, name), getattr(reduced.timings, name)
                )
        self.assertGreater(reduced.timings.total_seconds, 0.0)
        added = sum(
            getattr(completed.timings, name)
            for name in (
                "local_ionic_seconds",
                "initial_density_seconds",
                "core_density_seconds",
                "ion_ion_seconds",
            )
        )
        self.assertGreater(added, 0.0)
        # The reduced system is left as it was.
        self.assertIsNone(reduced.ionic_potential)
        self.assertTrue(np.isnan(reduced.ion_ion_energy))

        # The optional builders are the ones of the preparation.
        calls = []

        def local(*args):
            calls.append(("local", get_ident()))
            return build_local_ionic_potential(*args)

        def density(*args, core=False):
            calls.append(("core" if core else "valence", get_ident()))
            return superpose_atomic_density(*args, core=core)

        with_builders = complete_single_point(
            reduced, local_ionic_builder=local, atomic_density_builder=density
        )
        self.assertEqual(
            calls,
            [("local", get_ident()), ("valence", get_ident()), ("core", get_ident())],
        )
        assert_same_fields(with_builders)

        with self.assertRaisesRegex(ValueError, "already has the fields"):
            complete_single_point(complete)

        # The completed system is a complete one for the SCF loop.
        expected = run_scf(complete)
        actual = run_scf(completed)
        self.assertEqual(actual.energies.total, expected.energies.total)
        self.assertEqual(actual.density.tobytes(), expected.density.tobytes())

    def test_nlcc_density_enters_xc_but_is_kept_separate(self) -> None:
        path = Path(__file__).parent / "data" / "H_CORE_POTRE.DAT"
        problem = SinglePointInput(
            atoms=[Atom("H", [0.0, 0.0, 0.0])],
            pseudopotentials={"H": SpeciesPotential(path, 0)},
            grid=GridSettings(spacing=0.8, radius=3.0, expansion_order=2),
            scf=SCFSettings(max_iterations=1, number_of_states=2),
            eigensolver=EigensolverSettings(method="chebff"),
        )
        system = prepare_single_point(problem)
        self.assertGreater(system.grid.integrate(system.core_density), 0.0)
        with_core = system.evaluate_xc(system.initial_density)
        without_core = ca_lda(system.initial_density, system.grid.volume_element)
        self.assertGreater(
            float(np.max(np.abs(with_core.potential - without_core.potential))),
            1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
