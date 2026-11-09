"""Meta-GGAs of periodic cells (reference path): the Bloch terms and the k-point loops.

Independent references:

* the orbital terms of a Bloch Hamiltonian, ``(G + ik)^H vtau (G + ik)`` and in
  the orbital form of the density gradient ``2 sum_c [W_c G_c + G_c^T W_c]``,
  equal their dense matrices built from the stencil and are Hermitian;
* with the local potential they are the derivative of the discrete energy
  ``E_xc[n(u), tau(u), g(u)]`` with respect to the Bloch factors (central
  differences along a direction), on an orthorhombic and an oblique cell;
* the device Bloch operator on the NumPy backend applies the same terms;
* the space-group average of the orbital gradient returns the gradient of a
  symmetric density on a cubic grid (whose stencil the group maps onto itself);
* the Gamma-point loop is the 1x1x1 Bloch loop; two equal spin channels give
  the unpolarized k-point loop; the total energy of a converged k-point loop
  is the functional evaluated from its orbitals; both forms of the density
  gradient.
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from parsec_python import PeriodicCell, PeriodicGridSettings
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from parsec_python.Input import parse_parsec_input
from parsec_python.Laplacian import build_gradient, build_negative_laplacian
from parsec_python.SCF import prepare_periodic_single_point, run_scf
from parsec_python.SCF.kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized
from parsec_python.SCF.meta_gga_periodic import bloch_orbital_fields, symmetrized_gradient
from parsec_python.V_ion import NonlocalProjectorOperator
from parsec_python.V_xc.meta_gga import meta_gga_xc
from parsec_python.tests.unit.test_meta_gga import _LaplacianTestFunctional

DATA = Path(__file__).resolve().parents[1] / "data"


class _SmoothFunctional(_LaplacianTestFunctional):
    """The smooth test meta-GGA of ``test_meta_gga`` without its Laplacian term."""

    needs_laplacian = False

    def unpolarized(self, rho, sigma, tau, lapl):
        energy, vrho, vsigma, vtau, _ = super().unpolarized(rho, sigma, tau, np.zeros_like(rho))
        return energy, vrho, vsigma, vtau, None

    def polarized(self, *values):
        values = list(values)
        values[7] = np.zeros_like(values[0])
        values[8] = np.zeros_like(values[0])
        outputs = list(super().polarized(*values))
        outputs[8] = outputs[9] = None
        return tuple(outputs)


def _grids():
    """``(name, grid, lattice)``: Andrew Grieder's orthorhombic grid and Fortran's general grid of an fcc cell."""

    lattice = np.diag([4.2, 4.8, 5.4])
    orthorhombic = build_periodic_grid(
        PeriodicCell(lattice_vectors=lattice), PeriodicGridSettings(spacing=0.6, expansion_order=6)
    )
    fcc = 3.0 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    oblique = build_periodic_grid(
        PeriodicCell(lattice_vectors=fcc),
        PeriodicGridSettings(spacing=0.6, expansion_order=6, fortran_routines=True),
    )
    return (("orthorhombic", orthorhombic, lattice), ("oblique", oblique, fcc))


def _empty_projectors(size: int) -> NonlocalProjectorOperator:
    return NonlocalProjectorOperator(sp.csc_matrix((size, 0)), np.zeros(0), ())


def _bloch_factors(grid, lattice, count: int, seed: int) -> np.ndarray:
    """Smooth periodic complex factors (Euclidean-orthonormal) with a density bounded away from zero."""

    phase = 2.0 * np.pi * (grid.coordinates @ np.linalg.inv(lattice))
    rng = np.random.default_rng(seed)
    columns = []
    for index in range(count):
        coefficients = rng.uniform(-0.3, 0.3, size=6)
        column = (
            1.0
            + coefficients[0] * np.cos(phase[:, 0])
            + coefficients[1] * np.sin(phase[:, 1])
            + coefficients[2] * np.cos(phase[:, 2] + phase[:, 0])
            + 1j * (coefficients[3] * np.sin(phase[:, 0]) + coefficients[4] * np.cos(phase[:, 1] - phase[:, 2]))
            + 0.1 * index * np.exp(1j * phase[:, 2])
        )
        columns.append(column * (1.0 + coefficients[5] * np.sin(phase[:, 1])))
    q, _ = np.linalg.qr(np.column_stack(columns))
    return q


def _dense_terms(gradient, k_point, vtau, flux):
    """``sum_c D_c^H vtau D_c (+ 2 sum_c [W_c G_c + G_c^T W_c])`` as a dense matrix, ``D_c = G_c + i k_c``."""

    size = gradient[0].shape[0]
    total = np.zeros((size, size), dtype=np.complex128)
    for axis, matrix in enumerate(gradient):
        dense = matrix.toarray()
        derivative = dense + 1j * k_point[axis] * np.eye(size)
        total += derivative.conj().T @ (vtau[:, None] * derivative)
        if flux is not None:
            total += 2.0 * (flux[axis][:, None] * dense + dense.T @ np.diag(flux[axis]))
    return total


class BlochTermTests(unittest.TestCase):
    def test_terms_are_their_dense_matrices_and_hermitian(self) -> None:
        for name, grid, _ in _grids():
            with self.subTest(grid=name):
                gradient = build_gradient(grid)
                laplacian = build_negative_laplacian(grid)
                size = grid.size
                rng = np.random.default_rng(5)
                k_point = np.array([0.21, -0.13, 0.34])
                vtau = 0.4 + 0.2 * rng.random(size)
                flux = tuple(0.1 * rng.standard_normal(size) for _ in range(3))
                block = rng.standard_normal((size, 3)) + 1j * rng.standard_normal((size, 3))
                zero = np.zeros(size)
                for fluxes in (None, flux):
                    base = KPointKohnShamHamiltonian(laplacian, gradient, zero, _empty_projectors(size), k_point)
                    meta = KPointKohnShamHamiltonian(
                        laplacian, gradient, zero, _empty_projectors(size), k_point,
                        kinetic_potential=vtau, gradient_flux=fluxes,
                    )
                    term = meta.apply(block) - base.apply(block)
                    dense = _dense_terms(gradient, k_point, vtau, fluxes)
                    np.testing.assert_allclose(term, dense @ block, rtol=0, atol=1e-10 * np.abs(dense @ block).max())
                    self.assertLess(np.abs(dense - dense.conj().T).max(), 1e-11 * np.abs(dense).max())

    def test_terms_and_potential_are_the_derivative_of_the_energy(self) -> None:
        functional = _SmoothFunctional()
        weights = np.array([1.0, 0.8, 0.4])
        for name, grid, lattice in _grids():
            for orbital in (False, True):
                with self.subTest(grid=name, orbital_gradient=orbital):
                    gradient = tuple(build_gradient(grid))
                    laplacian = build_negative_laplacian(grid)
                    volume = grid.volume_element
                    k_point = np.array([0.19, 0.07, -0.23])

                    def evaluate(u):
                        fields = bloch_orbital_fields(
                            _System(grid, gradient), k_point, u, weights, 1.0, 2.0, with_gradient=orbital
                        )
                        density = (2.0 / volume) * np.sum(weights * (u.real ** 2 + u.imag ** 2), axis=1)
                        return meta_gga_xc(
                            functional,
                            density,
                            fields["tau"],
                            grid,
                            density_gradient=fields.get("gradient"),
                        )

                    u = _bloch_factors(grid, lattice, 3, seed=11)
                    result = evaluate(u)
                    hamiltonian = KPointKohnShamHamiltonian(
                        laplacian, gradient, result.potential, _empty_projectors(grid.size), k_point,
                        kinetic_potential=result.kinetic_potential,
                        gradient_flux=None if not orbital else tuple(result.gradient_flux),
                    )
                    # Only the XC terms: remove the kinetic stencil and the k terms of the base operator.
                    plain = KPointKohnShamHamiltonian(
                        laplacian, gradient, np.zeros(grid.size), _empty_projectors(grid.size), k_point
                    )
                    action = hamiltonian.apply(u) - plain.apply(u)
                    rng = np.random.default_rng(7)
                    direction = 1e-2 * (rng.standard_normal(u.shape) + 1j * rng.standard_normal(u.shape)) * np.abs(u)
                    step = 1e-3
                    plus = evaluate(u + step * direction).total_energy
                    minus = evaluate(u - step * direction).total_energy
                    plus2 = evaluate(u + 2.0 * step * direction).total_energy
                    minus2 = evaluate(u - 2.0 * step * direction).total_energy
                    numeric = (8.0 * (plus - minus) - (plus2 - minus2)) / (12.0 * step)
                    analytic = float(4.0 * np.sum(weights * np.real(np.einsum("ij,ij->j", direction.conj(), action))))
                    self.assertAlmostEqual(numeric, analytic, delta=1e-9)

    def test_device_operator_on_numpy_is_the_host_operator(self) -> None:
        from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter, NumpyArrayBackend

        for name, grid, _ in _grids():
            with self.subTest(grid=name):
                gradient = build_gradient(grid)
                laplacian = build_negative_laplacian(grid)
                rng = np.random.default_rng(9)
                size = grid.size
                k_point = np.array([0.0, 0.3, -0.1])
                block = rng.standard_normal((size, 4)) + 1j * rng.standard_normal((size, 4))
                potential = rng.standard_normal(size)
                for fluxes in (None, tuple(0.1 * rng.standard_normal(size) for _ in range(3))):
                    host = KPointKohnShamHamiltonian(
                        laplacian, gradient, potential, _empty_projectors(size), k_point,
                        kinetic_potential=0.5 + rng.random(size), gradient_flux=fluxes,
                    )
                    device = DeviceBlochConverter(NumpyArrayBackend()).operator(host)
                    np.testing.assert_allclose(
                        device.apply(np.ascontiguousarray(block)), host.apply(block), rtol=0, atol=1e-11
                    )

    def test_spectral_margin_bounds_a_vtau_spike(self) -> None:
        """Gershgorin's margin of the Bloch terms bounds them, reaches the eigensolvers and the device bound."""

        from parsec_python.Eigensolvers.spectral_bounds import orbital_term_margin
        from parsec_python.acceleration.Eigensolvers.chebff_spinor import (
            _CountingOperator,
            _lanczos_upper_bound,
            _Stats,
        )
        from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter, NumpyArrayBackend

        for name, grid, _ in _grids():
            with self.subTest(grid=name):
                gradient = build_gradient(grid)
                laplacian = build_negative_laplacian(grid)
                size = grid.size
                rng = np.random.default_rng(12)
                k_point = np.array([0.23, -0.17, 0.08])
                vtau = 0.3 + 0.1 * rng.random(size)
                vtau[size // 3] = 400.0  # a narrow spike, as in the vacuum of a slab
                flux = tuple(0.05 * rng.standard_normal(size) for _ in range(3))
                potential = 0.2 * rng.standard_normal(size)
                hamiltonian = KPointKohnShamHamiltonian(
                    laplacian, gradient, potential, _empty_projectors(size), k_point,
                    kinetic_potential=vtau, gradient_flux=flux,
                )
                terms = _dense_terms(gradient, k_point, vtau, flux)
                term_top = float(np.max(np.abs(np.linalg.eigvalsh(terms))))
                margin = hamiltonian.spectral_margin
                self.assertGreaterEqual(margin, term_top - 1e-9)
                self.assertEqual(orbital_term_margin(hamiltonian.as_linear_operator()), margin)
                device = DeviceBlochConverter(NumpyArrayBackend()).operator(hamiltonian)
                self.assertEqual(device.spectral_margin, margin)
                self.assertEqual(orbital_term_margin(device.as_linear_operator()), margin)
                # the device eigensolver's bound covers the whole spectrum despite the spike
                dense = (
                    laplacian.toarray()
                    + 1j * 0.0
                    + np.diag(potential + float(k_point @ k_point))
                    - 2j * sum(k_point[axis] * gradient[axis].toarray() for axis in range(3))
                    + terms
                )
                top = float(np.max(np.linalg.eigvalsh(dense)))
                counted = _CountingOperator(device, _Stats(None))  # the wrapper of the device solve
                upper, _ = _lanczos_upper_bound(np, counted, size, 6, np.random.default_rng(1))
                self.assertIsNotNone(device.kinetic)  # restored after the bare run
                self.assertGreaterEqual(upper, top)
                bare = KPointKohnShamHamiltonian(laplacian, gradient, potential, _empty_projectors(size), k_point)
                self.assertEqual(bare.spectral_margin, 0.0)

    def test_restart_run_of_a_periodic_meta_gga_is_refused(self) -> None:
        from parsec_python.Input import UnsupportedParsecOptionError

        directory = Path(tempfile.mkdtemp(prefix="parsec_meta_gga_restart_"))
        try:
            shutil.copy(DATA / "Si_FHIPP.DAT", directory / "Si_FHIPP.DAT")
            path = directory / "parsec.in"
            path.write_text(SILICON.format(extra="Restart_Run: .true.", criterion=1e-6))
            with self.assertRaises(UnsupportedParsecOptionError) as caught:
                parse_parsec_input(path)
            self.assertIn("Restart_Run", str(caught.exception))
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_symmetrized_gradient_keeps_the_gradient_of_a_symmetric_density(self) -> None:
        from parsec_python.Grid.space_group import find_space_group, periodic_grid_symmetry
        from parsec_python.models import Atom

        side = 4.8
        lattice = side * np.eye(3)
        grid = build_periodic_grid(PeriodicCell(lattice_vectors=lattice), PeriodicGridSettings(spacing=0.6, expansion_order=6))
        atoms = [Atom("Si", [0.0, 0.0, 0.0]), Atom("Si", [0.5 * side, 0.5 * side, 0.5 * side])]
        group = find_space_group(lattice, atoms)
        symmetry = periodic_grid_symmetry(grid, group)
        self.assertGreater(len(symmetry.images), 8)
        rng = np.random.default_rng(2)
        density = rng.random(grid.size)
        symmetric = np.mean([density[images] for images in symmetry.images], axis=0)
        values = np.stack([matrix @ symmetric for matrix in build_gradient(grid)])
        np.testing.assert_allclose(symmetrized_gradient(values, symmetry, lattice), values, rtol=0, atol=1e-12)
        noisy = values + 0.1 * rng.standard_normal(values.shape)
        once = symmetrized_gradient(noisy, symmetry, lattice)
        np.testing.assert_allclose(symmetrized_gradient(once, symmetry, lattice), once, rtol=0, atol=1e-12)


class _System:
    """The two members of a periodic system that :func:`bloch_orbital_fields` reads."""

    def __init__(self, grid, gradient) -> None:
        self.grid = grid
        self.gradient = gradient


SILICON = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  0.0 5.13 5.13
  5.13 0.0 5.13
  5.13 5.13 0.0
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.6
Expansion_Order: 6
Correlation_Type: scan
States_Num: 6
Fermi_Temp: 10 K
Convergence_Criterion: {criterion} Ry
Diag_Tolerance: 1e-8
Max_Iter: 60
{extra}
Atom_Types_Num: 1

Atom_Type: Si
Pseudopotential_Format: fhipp
Potential_Num: 3
Local_Component: d
Core_Cutoff_Radius: 2.2
begin Electron_Per_Orbital
2 2 0
end Electron_Per_Orbital
begin Atom_Coord
  0.0 0.0 0.0
  2.565 2.565 2.565
end Atom_Coord
"""

TWO_KPOINTS = """Kpoint_Method: manual
Kpoint_Unit: reciprocal_lattice_vectors
begin Kpoint_List
  0.0 0.0 0.0 1.0
  0.5 0.5 0.5 1.0
end Kpoint_List"""


SLAB = """Periodic_System: .true.
Boundary_Conditions: {boundary}
Use_Fortran_Routines: .{fortran}.
begin Cell_Shape
{cell}
end Cell_Shape
Lattice_Vector_Scale: 1
Boundary_Sphere_Radius: 5
Grid_Spacing: 0.55
Expansion_Order: 8
Atom_Types_Num: 1
Atom_Type: H
Element_Symbol: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
  1.0 1.0 0.0
  1.0 1.3 1.4
end Atom_Coord
Correlation_Type: scan
States_Num: 3
Convergence_Criterion: 1e-8 Ry
Max_Iter: 60
"""


class ConfinedLoopTests(unittest.TestCase):
    """Slabs (both sets of routines) and a wire: the Gamma loop is the 1x1x1 Bloch loop (orbital form)."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="parsec_meta_gga_confined_"))
        shutil.copy(DATA / "H_POTRE.DAT", self.directory / "H_POTRE.DAT")
        self.environment = mock.patch.dict(
            os.environ,
            {"PARSEC_KPOINT_BACKEND": "cpu", "PARSEC_KPOINT_WORKERS": "1", "PARSEC_META_GGA_GRADIENT": "orbital"},
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_gamma_loop_is_the_one_point_bloch_loop(self) -> None:
        cases = (
            ("slab", "false", "  5.0 0.0 0\n  2.0 5.0 0"),
            ("slab", "true", "  5.0 0.0 0\n  2.0 5.0 0"),
            ("wire", "false", "  4.0"),
        )
        for boundary, fortran, cell in cases:
            with self.subTest(boundary=boundary, fortran_routines=fortran):
                path = self.directory / f"{boundary}_{fortran}.in"
                path.write_text(SLAB.format(boundary=boundary, fortran=fortran, cell=cell))
                system = prepare_periodic_single_point(parse_parsec_input(path).problem)
                gamma = run_scf(system)
                bloch = run_scf_kpoints(system, np.zeros((1, 3)), np.ones(1))
                self.assertTrue(gamma.converged and bloch.converged)
                self.assertAlmostEqual(gamma.energies.total, bloch.energies.total, delta=1e-8)


class PeriodicLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="parsec_meta_gga_periodic_"))
        shutil.copy(DATA / "Si_FHIPP.DAT", self.directory / "Si_FHIPP.DAT")
        self.environment = mock.patch.dict(os.environ, {"PARSEC_KPOINT_BACKEND": "cpu", "PARSEC_KPOINT_WORKERS": "1"})
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        shutil.rmtree(self.directory, ignore_errors=True)

    def system(self, *, extra="", criterion=1e-7, name="parsec.in"):
        path = self.directory / name
        path.write_text(SILICON.format(extra=extra, criterion=criterion))
        return prepare_periodic_single_point(parse_parsec_input(path).problem)

    def test_gamma_loop_is_the_one_point_bloch_loop(self) -> None:
        for form in ("density", "orbital"):
            with self.subTest(gradient=form), mock.patch.dict(os.environ, {"PARSEC_META_GGA_GRADIENT": form}):
                system = self.system()
                gamma = run_scf(system)
                bloch = run_scf_kpoints(system, np.zeros((1, 3)), np.ones(1))
                self.assertTrue(gamma.converged and bloch.converged)
                self.assertEqual(gamma.meta_gga.gradient, form)
                self.assertAlmostEqual(gamma.energies.total, bloch.energies.total, delta=1e-9)
                self.assertAlmostEqual(
                    gamma.energies.integral_vtau_tau, bloch.energies.integral_vtau_tau, delta=1e-6
                )

    def test_two_equal_spin_channels_are_the_unpolarized_kpoint_loop(self) -> None:
        # A bulk cell takes the smart SUBSPACE lower bound by default since 2026-11-08; with it the spin loop of
        # this cell stalls at a residual of ~1.5e-7 Ry (54 iterations on a laptop, more than Max_Iter 60 on an
        # A100; 28 with PARSEC 1.5's adaptive rule or the earlier monotone one, the same energy).  The test is
        # about the two equal channels, so it names the adaptive rule for both runs.
        rule = "\nSmart_Subspace: .false."
        system = self.system(extra=TWO_KPOINTS + rule)
        # The unpolarized cell removes the average of V_xc by default (PARSEC 1.5's potshift, since 2026-11-08), a
        # spin-polarized one does not: the spin run asks for it, so that its two equal channels carry the potential of
        # the unpolarized loop.  Without it the spin loop's spectrum sits ~1 Ry lower, its largest Ritz value near
        # 0 Ry, where the adaptive bound's ratio of Ritz values to the previous bound jumps, and the loop stalls
        # (150 iterations, the same energy to 1e-11 Ry; 52 with the default smart bound).
        spin = self.system(
            extra=TWO_KPOINTS + rule + "\nSpin_Polarization: .true.\nZero_Average_Vxc: .true.", name="spin.in"
        )
        from parsec_python.Grid import kpoint_sampling

        k_points, weights = kpoint_sampling(system.input)
        unpolarized = run_scf_kpoints(system, k_points, weights)
        polarized = run_scf_kpoints_spin_polarized(spin, k_points, weights)
        self.assertTrue(unpolarized.converged and polarized.converged)
        self.assertAlmostEqual(unpolarized.energies.total, polarized.energies.total, delta=5e-8)
        np.testing.assert_allclose(
            polarized.meta_gga.kinetic_energy_densities[0], polarized.meta_gga.kinetic_energy_densities[1], atol=1e-6
        )

    def test_energy_is_the_functional_of_the_orbitals(self) -> None:
        from parsec_python.Grid import kpoint_sampling
        from parsec_python.V_ion import build_nonlocal_projectors

        for functional, form in (("pbe", "density"), ("scan", "density"), ("scan", "orbital")):
            with self.subTest(functional=functional, gradient=form), mock.patch.dict(
                os.environ, {"PARSEC_META_GGA_GRADIENT": form}
            ):
                system = self.system(extra=TWO_KPOINTS, criterion=1e-9, name=f"{functional}_{form}.in")
                if functional == "pbe":
                    from dataclasses import replace

                    scf = replace(system.input.scf, xc_functional="pbe")
                    system = replace(system, input=replace(system.input, scf=scf))
                k_points, weights = kpoint_sampling(system.input)
                result = run_scf_kpoints(system, k_points, weights, keep_wavefunctions=True)
                self.assertTrue(result.converged)
                volume = system.grid.volume_element
                states = len(result.eigenvalues) // len(k_points)
                density = np.zeros(system.grid.size)
                tau = np.zeros(system.grid.size)
                orbital_gradient = np.zeros((3, system.grid.size))
                one_body = 0.0
                for index, (k_point, weight) in enumerate(zip(k_points, weights)):
                    u = result.k_wavefunctions[index]
                    occupations = result.occupations[index * states : (index + 1) * states]
                    projectors = build_nonlocal_projectors(
                        system.grid, system.cell_atoms, system.pseudopotentials, system.input.pseudopotentials,
                        system.input.periodic_cell.lattice_vectors, k_point=k_point,
                        periodic_dimensions=system.input.periodic_cell.dimensions,
                    )
                    kinetic = KPointKohnShamHamiltonian(
                        system.negative_laplacian, system.gradient, np.zeros(system.grid.size), projectors, k_point
                    )
                    action = kinetic.apply(u)
                    one_body += 2.0 * weight * float(
                        np.sum(occupations * np.real(np.einsum("ij,ij->j", u.conj(), action)))
                    )
                    density += weight * (2.0 / volume) * np.sum(occupations * np.abs(u) ** 2, axis=1)
                    fields = bloch_orbital_fields(system, k_point, u, occupations, weight, 2.0, with_gradient=True)
                    tau += fields["tau"]
                    orbital_gradient += fields["gradient"]
                hartree = system.solve_hartree(density).potential
                if functional == "pbe":
                    xc_energy = system.evaluate_xc(density).total_energy
                else:
                    xc_energy = system.evaluate_xc(
                        density, tau, density_gradient=None if form == "density" else tuple(orbital_gradient)
                    ).total_energy
                direct = (
                    one_body
                    + volume * float(np.dot(density, system.ionic_potential))
                    + 0.5 * volume * float(np.dot(density, hartree))
                    + xc_energy
                    + system.ion_ion_energy
                    + system.alpha_z_energy
                )
                self.assertAlmostEqual(result.energies.total, direct, delta=2e-6)


if __name__ == "__main__":
    unittest.main()
