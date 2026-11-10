"""The meta-GGA orbital term in the symmetry sectors and on the devices.

* the sector-reduced derivatives ``B_{Gamma,a}`` give ``U^T H_tau U`` of the
  full-grid term at round-off, and ``tau`` of sector orbitals is that of their
  full-grid expansions (reflection groups, a signed-permutation group, grids
  with points on the mirror planes);
* on a device: the term of a CuPy operator is the host term, in ``apply``,
  ``apply_into`` and the Chebyshev recurrence, and ``tau`` of device orbitals
  is the host ``tau``;
* the orbital form of the density gradient: the first-order term of ``W`` in
  the sectors and on the full grid, ``g`` of sector orbitals, the vector
  fields at the orbit representatives, the device functional and the host one
  on orbits; Gershgorin's bound of the term holds its eigenvalues.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.V_xc.meta_gga import KineticPotentialOperator, grid_gradient, kinetic_energy_density
from parsec_python.acceleration.Hamiltonian.meta_gga_device import (
    derivative_directions,
    sector_derivatives,
)
from parsec_python.acceleration.Symmetry import (
    AxisReflectionReduction,
    ReflectionRepresentationDecomposition,
    SignedPermutationReduction,
)
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.models import Atom, GridSettings

MOLECULES = {
    "D2h": [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    "C2v": [("O", (0.0, 0.0, 0.0)), ("H", (1.43, 0.0, 1.1)), ("H", (-1.43, 0.0, 1.1))],
    # Mirror planes x = y and x = -y and the z axis (signed permutations).
    "signed": [("C", (0.0, 0.0, 0.0)), ("H", (1.1, 1.1, 0.7)), ("H", (-1.1, -1.1, 0.7))],
}


def _grid(shift):
    return build_cluster_grid(GridSettings(spacing=0.55, radius=3.3, expansion_order=8, shift=shift))


def _decomposition(grid, molecule, signed=False):
    atoms = [Atom(symbol, position) for symbol, position in MOLECULES[molecule]]
    detect = SignedPermutationReduction.detect if signed else AxisReflectionReduction.detect
    reduction = detect(grid, atoms)
    return ReflectionRepresentationDecomposition.build(grid, reduction)


def _invariant(grid, decomposition, seed=1):
    """A positive field with the symmetry of the sectors."""

    rng = np.random.default_rng(seed)
    wedge = 0.2 + rng.random(decomposition.wedge_size)
    return wedge[decomposition.reduction.full_to_wedge]


def _cases():
    for shift in ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0)):
        grid = _grid(shift)
        for molecule, signed in (("D2h", False), ("C2v", False), ("signed", True)):
            yield f"{molecule} shift {shift[0]}", grid, _decomposition(grid, molecule, signed)


class SectorReductionTests(unittest.TestCase):
    def test_directions_are_common_eigenvectors(self) -> None:
        grid = _grid((0.5, 0.5, 0.5))
        decomposition = _decomposition(grid, "signed", True)
        directions, characters = derivative_directions(decomposition.reduction)
        np.testing.assert_allclose(directions.T @ directions, np.eye(3), atol=1e-14)
        self.assertTrue(np.all(np.abs(characters) == 1.0))
        # Not the Cartesian axes: the group mixes x and y.
        self.assertFalse(np.allclose(np.abs(directions), np.eye(3)))

    def test_sector_term_and_tau_are_those_of_the_full_grid(self) -> None:
        for label, grid, decomposition in _cases():
            with self.subTest(label):
                vtau = _invariant(grid, decomposition)
                full = KineticPotentialOperator(grid_gradient(grid), vtau)
                derivatives = sector_derivatives(grid, decomposition)
                rng = np.random.default_rng(4)
                tau_full = np.zeros(grid.size)
                tau_wedge = np.zeros(decomposition.wedge_size)
                multiplicities = decomposition.reduction.multiplicities.astype(float)
                for representation, pieces in derivatives.items():
                    size = decomposition.sector_size(representation)
                    block = rng.standard_normal((size, 2))
                    weights = np.array([1.0, 0.4])
                    reduced = np.zeros_like(block)
                    for piece in pieces:
                        orbits = decomposition.sector_orbit_indices(piece.target)
                        values = piece.matrix @ block
                        reduced += piece.matrix.T @ (vtau[decomposition.reduction.representative_rows[orbits]][:, None] * values)
                        tau_wedge[orbits] += (values * values) @ weights / multiplicities[orbits]
                    expanded = np.column_stack([decomposition.expand_vector(column, representation) for column in block.T])
                    applied = full.apply(expanded)
                    expected = np.column_stack(
                        [decomposition.reduce_vector(column, representation) for column in applied.T]
                    )
                    np.testing.assert_allclose(reduced, expected, rtol=0, atol=1e-11 * np.max(np.abs(expected)))
                    tau_full += kinetic_energy_density(expanded, weights, grid_gradient(grid), 1.0, degeneracy=2.0)
                # kinetic_energy_density holds 1/2 * degeneracy 2 = 1.
                np.testing.assert_allclose(
                    tau_wedge[decomposition.reduction.full_to_wedge], tau_full, rtol=1e-11, atol=1e-12 * tau_full.max()
                )


@unittest.skipUnless(cupy_available(), "CuPy is not available")
class DeviceScanTests(unittest.TestCase):
    def test_scan_on_the_device_is_scan_on_the_host(self) -> None:
        # Negative zeta (a complex power of a negative base would carry an
        # imaginary part far above the complex step on CuPy) and far tails
        # (sigma ~ 1e-30, where an absolute step is no small step).
        import cupy as cp

        from parsec_python.V_xc.scan import NativeScan

        rng = np.random.default_rng(5)
        count = 4000
        total = 10 ** rng.uniform(-14, 1.5, count)
        share = rng.uniform(0.02, 0.98, count)
        up, down = total * share, total * (1.0 - share)
        s_uu = (up * rng.uniform(0, 4, count)) ** 2
        s_dd = (down * rng.uniform(0, 4, count)) ** 2
        s_ud = np.sqrt(s_uu * s_dd) * rng.uniform(-1, 1, count)
        t_up = s_uu / (8 * up) * rng.uniform(0.3, 3, count)
        t_down = s_dd / (8 * down) * rng.uniform(0.3, 3, count)
        host = NativeScan().polarized(up, down, s_uu, s_ud, s_dd, t_up, t_down)
        device = NativeScan().polarized(*(cp.asarray(value) for value in (up, down, s_uu, s_ud, s_dd, t_up, t_down)))
        for expected, value in zip(host[:8], device[:8]):
            value = cp.asnumpy(value)
            self.assertTrue(np.all(np.isfinite(value)))
            scale = np.abs(expected) + 1e-10 * np.max(np.abs(expected))
            np.testing.assert_array_less(np.abs(value - expected) / scale, 1e-8)
        rho, sigma, tau = total, s_uu + 2 * s_ud + s_dd, t_up + t_down
        host = NativeScan().unpolarized(rho, sigma, tau)
        device = NativeScan().unpolarized(cp.asarray(rho), cp.asarray(sigma), cp.asarray(tau))
        for expected, value in zip(host[:4], device[:4]):
            value = cp.asnumpy(value)
            scale = np.abs(expected) + 1e-10 * np.max(np.abs(expected))
            np.testing.assert_array_less(np.abs(value - expected) / scale, 1e-8)


@unittest.skipUnless(cupy_available(), "CuPy is not available")
class DeviceTermTests(unittest.TestCase):
    def test_device_operator_holds_the_host_term(self) -> None:
        import cupy as cp
        import scipy.sparse as sp

        from parsec_python.Laplacian import build_negative_laplacian
        from parsec_python.acceleration.Hamiltonian.meta_gga_device import DeviceKineticTerm
        from parsec_python.acceleration.backends.cupy import CuPyHamiltonian

        grid = _grid((0.5, 0.5, 0.5))
        laplacian = build_negative_laplacian(grid)
        potential = -0.3 * np.exp(-0.2 * np.sum(grid.coordinates**2, axis=1))
        projectors = sp.csr_matrix((grid.size, 0))
        operator = CuPyHamiltonian(laplacian, potential, projectors=projectors, projector_signs=np.empty(0))
        vtau = 0.1 + 0.2 * np.exp(-0.3 * np.sum(grid.coordinates**2, axis=1))
        derivatives = sector_derivatives(grid, None)[None]
        term = DeviceKineticTerm(cp, derivatives, [None, None, None], 0)
        term.bind(cp.asarray(vtau))
        operator.bind_kinetic_term(term)
        host = KineticPotentialOperator(grid_gradient(grid), vtau)
        rng = np.random.default_rng(2)
        block = rng.standard_normal((grid.size, 5))
        expected = laplacian @ block + potential[:, None] * block + host.apply(block)
        for order in ("C", "F"):
            device_block = cp.asarray(block, order=order)
            np.testing.assert_allclose(cp.asnumpy(operator.apply(device_block)), expected, rtol=0, atol=1e-10)
            output = cp.empty_like(device_block)
            operator.apply_into(device_block, output)
            np.testing.assert_allclose(cp.asnumpy(output), expected, rtol=0, atol=1e-10)
        previous = rng.standard_normal((grid.size, 5))
        recurrence = operator.chebyshev_recurrence(
            cp.asarray(block), center=0.4, scale=0.3, sigma_next=0.9, previous=cp.asarray(previous), sigma=0.2
        )
        reference = 0.9 * (0.3 * (expected - 0.4 * block) - 0.2 * previous)
        np.testing.assert_allclose(cp.asnumpy(recurrence), reference, rtol=0, atol=1e-10)
        weights = np.array([1.0, 1.0, 0.5, 0.2, 0.0])
        sums = term.kinetic_sums(cp.asarray(block), weights)
        tau_device = 0.5 * sum(cp.asnumpy(value) for value in sums)
        tau_host = kinetic_energy_density(block, weights, grid_gradient(grid), 1.0, degeneracy=1.0)
        np.testing.assert_allclose(tau_device, tau_host, rtol=1e-12, atol=1e-12)
        operator.bind_kinetic_term(None)
        np.testing.assert_allclose(
            cp.asnumpy(operator.apply(cp.asarray(block))), laplacian @ block + potential[:, None] * block, atol=1e-10
        )


def _gradient_field(grid, decomposition, seed=3):
    """An equivariant vector field: the stencil gradient of an invariant field, ``(3, n)``."""

    return grid_gradient(grid).apply(_invariant(grid, decomposition, seed))


class OrbitVectorFieldTests(unittest.TestCase):
    def test_representatives_hold_the_vector_field(self) -> None:
        from parsec_python.acceleration.Hamiltonian.meta_gga_device import OrbitVectorFields

        for label, grid, decomposition in _cases():
            with self.subTest(label):
                field = _gradient_field(grid, decomposition)
                fields = OrbitVectorFields(grid, decomposition.reduction)
                reduced = fields.reduce(field)
                self.assertEqual(reduced.shape, (3, decomposition.wedge_size))
                np.testing.assert_allclose(fields.expand(reduced), field, rtol=0, atol=1e-12 * np.max(np.abs(field)))

    def test_warm_start_w_crosses_between_the_sectors_and_the_full_grid(self) -> None:
        # A record holds W as full-grid Cartesian components; a loop on orbits
        # takes them at its representatives and gives them back, and a
        # full-grid loop (the fallback of the spin sectors after an ionic move)
        # takes the record of a sector loop as it is.
        from types import SimpleNamespace
        from unittest import mock

        from parsec_python.SCF.meta_gga import MetaGGALoop
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer

        grid = _grid((0.5, 0.5, 0.5))
        decomposition = _decomposition(grid, "C2v")
        reducer = SymmetrySCFReducer(decomposition.reduction)
        system = SimpleNamespace(input=SimpleNamespace(scf=SimpleNamespace(xc_functional="scan")), grid=grid)
        vtau = _invariant(grid, decomposition)
        flux = tuple(_gradient_field(grid, decomposition))
        start = SimpleNamespace(kinetic_potential=vtau, gradient_flux=(flux,))
        with mock.patch.dict("os.environ", {"PARSEC_META_GGA_GRADIENT": "orbital"}):
            sectors = MetaGGALoop(system, channels=1, adapter=reducer, warm_start=start)
            rows = decomposition.reduction.representative_rows
            for component, expected in zip(sectors.input_flux[0], flux):
                np.testing.assert_array_equal(reducer.wedge_values(component), expected[rows])
            record = sectors.record()
            for component, expected in zip(record.next_gradient_fluxes[0], flux):
                self.assertEqual(np.asarray(component).shape, (grid.size,))
                np.testing.assert_allclose(component, expected, rtol=0, atol=1e-14)
            full = MetaGGALoop(
                system,
                channels=1,
                warm_start=SimpleNamespace(
                    kinetic_potential=record.next_kinetic_potentials[0], gradient_flux=record.next_gradient_fluxes
                ),
            )
            for component, expected in zip(full.input_flux[0], flux):
                np.testing.assert_allclose(component, expected, rtol=0, atol=1e-14)

    def test_vector_symmetrization_keeps_turning_fields(self) -> None:
        # Symmetrize_Charge_Density averages g of the orbital form as a vector
        # field: a field that turns with the group is unchanged, any other
        # becomes one (the average is idempotent).
        from parsec_python.Grid.point_group import grid_point_group, symmetrize_vector

        grid = _grid((0.5, 0.5, 0.5))
        decomposition = _decomposition(grid, "D2h")
        matrices = np.stack([np.diag(signs) for signs in np.asarray(decomposition.reduction.signs, dtype=float)])
        group = grid_point_group(grid, matrices)
        turning = _gradient_field(grid, decomposition)
        np.testing.assert_allclose(symmetrize_vector(turning, group), turning, rtol=0, atol=1e-14)
        noise = np.random.default_rng(2).standard_normal((3, grid.size))
        averaged = symmetrize_vector(noise, group)
        np.testing.assert_allclose(symmetrize_vector(averaged, group), averaged, rtol=0, atol=1e-14)

    def test_gershgorin_bound_of_the_term(self) -> None:
        # The bound holds every eigenvalue of the term, with a spike of vtau
        # and the first-order term of the orbital form.
        grid = _grid((0.5, 0.5, 0.5))
        decomposition = _decomposition(grid, "D2h")
        gradient = grid_gradient(grid)
        vtau = 0.1 * _invariant(grid, decomposition)
        vtau[np.argmin(np.sum(grid.coordinates**2, axis=1))] = 40.0
        for flux in (None, 0.3 * _gradient_field(grid, decomposition)):
            term = KineticPotentialOperator(gradient, vtau, gradient_flux=flux)
            eigenvalues = np.linalg.eigvalsh(term.as_sparse().toarray())
            self.assertGreaterEqual(term.spectral_margin, np.max(np.abs(eigenvalues)))
            # Far below max(vtau) * ||sum_c G_c^T G_c||: a spike is local.
            stiffness = np.linalg.eigvalsh(
                KineticPotentialOperator(gradient, np.ones(grid.size)).as_sparse().toarray()
            )[-1]
            self.assertLess(term.spectral_margin, 0.6 * 40.0 * stiffness)


class FilterBoundTests(unittest.TestCase):
    def test_upper_bound_takes_in_the_term(self) -> None:
        from parsec_python.acceleration.Eigensolvers.spectral_bounds import with_orbital_term

        class Operator:
            kinetic_term = None
            spectral_margin = 0.0
            bare_upper_bound = None
            effective_potential = np.array([-3.0, 1.0])

        operator = Operator()
        # Without the term: the Lanczos bound, kept as the bare one.
        self.assertEqual(with_orbital_term(operator, 600.0), 600.0)
        self.assertEqual(operator.bare_upper_bound, (600.0, 1.0))
        operator.kinetic_term = object()
        operator.spectral_margin = 250.0
        # A Lanczos run that missed the term's localized eigenvalues: bare + m.
        self.assertEqual(with_orbital_term(operator, 605.0), 850.0)
        # The largest potential rose by 2 since the bare bound.
        operator.effective_potential = np.array([-3.0, 3.0])
        self.assertEqual(with_orbital_term(operator, 605.0), 852.0)
        # A Lanczos bound above bare + m stays.
        self.assertEqual(with_orbital_term(operator, 900.0), 900.0)
        # Before any bare bound: Lanczos + m.
        operator.bare_upper_bound = None
        self.assertEqual(with_orbital_term(operator, 605.0), 855.0)


@unittest.skipUnless(cupy_available(), "CuPy is not available")
class DeviceOrbitalGradientTests(unittest.TestCase):
    def test_sector_term_and_gradient_of_the_orbital_form(self) -> None:
        # 2 sum_a [W'_a D'_a + D'_a^T W'_a] in the sectors (shared rows of a
        # sector and its target) is U^T of the full-grid term, g of sector
        # orbitals that of their expansions, and the Gershgorin bound of each
        # sector term holds its eigenvalues.
        import cupy as cp

        from parsec_python.V_xc.meta_gga import orbital_density_gradient
        from parsec_python.acceleration.Hamiltonian.meta_gga_device import DeviceKineticTerm

        for label, grid, decomposition in _cases():
            with self.subTest(label):
                gradient = grid_gradient(grid)
                vtau = _invariant(grid, decomposition)
                flux = 0.4 * _gradient_field(grid, decomposition)
                full = KineticPotentialOperator(gradient, vtau, gradient_flux=flux)
                directions, _ = derivative_directions(decomposition.reduction)
                rows = decomposition.reduction.representative_rows
                along = directions.T @ flux[:, rows]
                multiplicities = decomposition.reduction.multiplicities.astype(float)
                rng = np.random.default_rng(6)
                weights = np.array([1.0, 0.4])
                g_along = np.zeros((3, decomposition.wedge_size))
                g_full = np.zeros((3, grid.size))
                for representation, pieces in sector_derivatives(grid, decomposition).items():
                    orbits = [decomposition.sector_orbit_indices(piece.target) for piece in pieces]
                    term = DeviceKineticTerm(
                        cp, pieces, orbits, 0, source_orbits=decomposition.sector_orbit_indices(representation)
                    )
                    term.bind(cp.asarray(vtau[rows]), [cp.asarray(row) for row in along])
                    size = decomposition.sector_size(representation)
                    block = rng.standard_normal((size, 2))
                    expanded = np.column_stack([decomposition.expand_vector(column, representation) for column in block.T])
                    expected = np.column_stack(
                        [decomposition.reduce_vector(column, representation) for column in full.apply(expanded).T]
                    )
                    applied = cp.asnumpy(term.apply(cp.asarray(block)))
                    np.testing.assert_allclose(applied, expected, rtol=0, atol=1e-11 * np.max(np.abs(expected)))
                    dense = np.column_stack([cp.asnumpy(term.apply(cp.asarray(column))) for column in np.eye(size)])
                    self.assertGreaterEqual(term.spectral_margin, np.max(np.abs(np.linalg.eigvalsh(dense))) - 1e-10)
                    _, products = term.kinetic_sums(cp.asarray(block), weights, gradient=True)
                    for axis, (shared, values) in enumerate(zip(term.shared_orbits, products)):
                        g_along[axis, shared] += cp.asnumpy(values) / multiplicities[shared]
                    g_full += orbital_density_gradient(expanded, weights, gradient, 1.0, degeneracy=1.0)
                np.testing.assert_allclose(
                    directions @ (2.0 * g_along), g_full[:, rows], rtol=0, atol=1e-11 * np.max(np.abs(g_full))
                )

    def test_full_grid_term_of_the_orbital_form(self) -> None:
        import cupy as cp

        from parsec_python.V_xc.meta_gga import orbital_density_gradient
        from parsec_python.acceleration.Hamiltonian.meta_gga_device import DeviceKineticTerm

        grid = _grid((0.5, 0.5, 0.5))
        decomposition = _decomposition(grid, "C2v")
        gradient = grid_gradient(grid)
        vtau = _invariant(grid, decomposition)
        vtau[7] = 25.0
        rng = np.random.default_rng(8)
        flux = rng.standard_normal((3, grid.size))
        host = KineticPotentialOperator(gradient, vtau, gradient_flux=flux)
        term = DeviceKineticTerm(cp, sector_derivatives(grid, None)[None], [None, None, None], 0)
        term.bind(cp.asarray(vtau), [cp.asarray(row) for row in flux])
        block = rng.standard_normal((grid.size, 4))
        np.testing.assert_allclose(cp.asnumpy(term.apply(cp.asarray(block))), host.apply(block), rtol=0, atol=1e-10)
        self.assertAlmostEqual(term.spectral_margin, host.spectral_margin, delta=1e-9 * host.spectral_margin)
        weights = np.array([1.0, 0.5, 0.25, 0.0])
        _, products = term.kinetic_sums(cp.asarray(block), weights, gradient=True)
        expected = orbital_density_gradient(block, weights, gradient, 1.0, degeneracy=1.0)
        np.testing.assert_allclose(2.0 * np.stack([cp.asnumpy(value) for value in products]), expected, atol=1e-12)
        term.bind(cp.asarray(vtau))
        np.testing.assert_allclose(
            cp.asnumpy(term.apply(cp.asarray(block))),
            KineticPotentialOperator(gradient, vtau).apply(block),
            rtol=0,
            atol=1e-10,
        )

    def test_targeted_bound_sees_the_spike(self) -> None:
        # A narrow spike of vtau gives the operator a localized top eigenvalue;
        # the targeted Lanczos bound (start on the term's row sums) is above it.
        from unittest import mock

        import cupy as cp
        import scipy.sparse as sp

        from parsec_python.Laplacian import build_negative_laplacian
        from parsec_python.acceleration.Eigensolvers.spectral_bounds import lanczos_upper_bound
        from parsec_python.acceleration.Hamiltonian.meta_gga_device import DeviceKineticTerm
        from parsec_python.acceleration.backends.cupy import CuPyHamiltonian

        grid = _grid((0.5, 0.5, 0.5))
        laplacian = build_negative_laplacian(grid)
        potential = -0.3 * np.exp(-0.2 * np.sum(grid.coordinates**2, axis=1))
        operator = CuPyHamiltonian(
            laplacian, potential, projectors=sp.csr_matrix((grid.size, 0)), projector_signs=np.empty(0)
        )
        vtau = 0.05 + 0.1 * np.exp(-0.3 * np.sum(grid.coordinates**2, axis=1))
        vtau[np.argmax(np.sum(grid.coordinates**2, axis=1) * (grid.coordinates[:, 0] > 0))] = 60.0
        term = DeviceKineticTerm(cp, sector_derivatives(grid, None)[None], [None, None, None], 0)
        term.bind(cp.asarray(vtau))
        operator.bind_kinetic_term(term)
        dense = laplacian.toarray() + np.diag(potential) + KineticPotentialOperator(grid_gradient(grid), vtau).as_sparse().toarray()
        top = float(np.linalg.eigvalsh(dense)[-1])
        for policy in ("targeted", "gershgorin"):
            with mock.patch.dict("os.environ", {"PARSEC_META_GGA_FILTER_BOUND": policy}):
                bound = lanczos_upper_bound(operator, steps=8, rng=np.random.default_rng(3))
            self.assertGreaterEqual(bound.upper_bound, top, policy)

    def test_device_floors_are_the_host_floors(self) -> None:
        # tau_W and the absolute floor of the grid layer on the device.
        from parsec_python.V_xc.meta_gga import meta_gga_xc
        from parsec_python.V_xc.scan import NativeScan
        from parsec_python.acceleration.V_xc.meta_gga_xc import DeviceMetaGGAOnOrbits

        grid = _grid((0.5, 0.5, 0.5))
        r2 = np.sum(grid.coordinates**2, axis=1)
        density = 0.8 * np.exp(-0.9 * r2) + 1e-9
        gradient = grid_gradient(grid).apply(density)
        tau = 0.7 * np.sum(gradient**2, axis=0) / (8.0 * density) + 0.01 * np.exp(-0.5 * r2)
        functional = NativeScan()
        functional.tau_floor = "weizsaecker"
        functional.tau_minimum = float(np.quantile(tau, 0.2))
        reference = meta_gga_xc(functional, density, tau, grid)
        result = DeviceMetaGGAOnOrbits(functional, grid)(density, tau=tau)
        self.assertAlmostEqual(result.total_energy, reference.total_energy, delta=1e-12)
        for left, right in ((result.potential, reference.potential), (result.kinetic_potential, reference.kinetic_potential)):
            np.testing.assert_allclose(left, right, rtol=0, atol=1e-11 * np.max(np.abs(right)))

    def test_device_functional_of_the_orbital_form(self) -> None:
        # On orbits of a reflection group and on the full grid, unpolarized
        # and spin, with a core density: the host functional of the full grid.
        import cupy as cp  # noqa: F401

        from parsec_python.V_xc.meta_gga import meta_gga_xc, meta_gga_xc_spin
        from parsec_python.V_xc.scan import NativeScan
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
        from parsec_python.acceleration.V_xc.meta_gga_xc import DeviceMetaGGAOnOrbits, HostMetaGGAOnOrbits

        grid = _grid((0.5, 0.5, 0.5))
        decomposition = _decomposition(grid, "C2v")
        reduction = decomposition.reduction
        rows = reduction.representative_rows
        r2 = np.sum((grid.coordinates - np.array([0.0, 0.0, 0.3])) ** 2, axis=1)
        density = 0.8 * np.exp(-0.9 * r2) + 1e-9
        core = 0.3 * np.exp(-2.5 * r2)
        gradient = grid_gradient(grid)
        orbital = 0.9 * gradient.apply(density)
        tau = 0.3 * np.exp(-0.7 * r2) + np.sum(orbital**2, axis=0) / (8.0 * density) * 1.05
        functional = NativeScan()
        reference = meta_gga_xc(functional, density, tau, grid, core, density_gradient=orbital)
        reducer = SymmetrySCFReducer(reduction)

        class Reference:
            def __init__(self):
                self.grid = grid

            def evaluate_xc(self, density, tau=None, density_gradient=None):
                return meta_gga_xc(
                    functional, density, tau, grid, core,
                    density_gradient=None if density_gradient is None else np.stack(density_gradient),
                )

        cases = (
            ("full grid", None, density, tau, tuple(orbital), lambda values: values),
            ("orbits", reducer, reducer.field(density[rows]), reducer.field(tau[rows]),
             tuple(reducer.field(np.ascontiguousarray(row)) for row in orbital[:, rows]), lambda values: values[..., rows]),
        )
        for label, given, n, t, g, at in cases:
            evaluators = (
                ("device", DeviceMetaGGAOnOrbits(functional, grid, reducer=given, core_density=core)),
                ("host", HostMetaGGAOnOrbits(lambda: Reference(), given)),
            )
            for name, evaluator in evaluators:
                with self.subTest(f"{label} {name}"):
                    result = evaluator(n, tau=t, density_gradient=g)
                    values = (lambda field: np.asarray(reducer.wedge_values(field))) if given is not None else np.asarray
                    self.assertAlmostEqual(result.total_energy, reference.total_energy, delta=1e-11)
                    for left, right in (
                        (result.potential, reference.potential),
                        (result.kinetic_potential, reference.kinetic_potential),
                        (result.core_potential, reference.core_potential),
                    ):
                        right = at(right)
                        np.testing.assert_allclose(values(left), right, rtol=0, atol=1e-10 * np.max(np.abs(right)))
                    flux = np.stack([values(component) for component in result.gradient_flux])
                    np.testing.assert_allclose(flux, at(reference.gradient_flux), rtol=0, atol=1e-10)
        # Spin: channels of 0.6 and 0.4 of the density, the full grid.
        up, down = 0.6 * density, 0.4 * density
        gradients = (0.6 * orbital, 0.4 * orbital)
        reference = meta_gga_xc_spin(functional, up, down, 0.6 * tau, 0.4 * tau, grid, core, density_gradients=gradients)
        device = DeviceMetaGGAOnOrbits(functional, grid, core_density=core, spin=True)
        result = device(up, down, tau_up=0.6 * tau, tau_down=0.4 * tau, density_gradients=tuple(tuple(g) for g in gradients))
        self.assertAlmostEqual(result.total_energy, reference.total_energy, delta=1e-11)
        for left, right in (
            (result.potential_up, reference.potential_up),
            (result.potential_down, reference.potential_down),
            (result.core_potential, reference.core_potential),
            (np.stack(result.gradient_flux_up), reference.gradient_flux_up),
            (np.stack(result.gradient_flux_down), reference.gradient_flux_down),
        ):
            np.testing.assert_allclose(left, right, rtol=0, atol=1e-10 * np.max(np.abs(right)))


if __name__ == "__main__":
    unittest.main()
