"""Coulomb kernels of the exact exchange in periodic cells (Hartree/exchange_kernel.py).

References used here, all independent of the module:

* Bloch sums of normalized Gaussians ``g_a = (a/pi)**1.5 exp(-a r**2)`` ("pair densities") whose k-averaged
  energy ``(1/N_k) sum_q dV sum_r conj(p_q) w_q`` tends to the isolated self-energy of the Gaussian:
  ``4 sqrt(b/pi)`` Ry for ``1/r`` and ``4 sqrt(c/pi)`` for ``erf(w r)/r`` (``b = a/2``, ``1/c = 1/b + 1/w**2``),
  erfc the difference;
* at a finite mesh the erfc energy with its exact ``q + G = 0`` limit is the real-space sum of the Gaussian with its
  Born-von Karman images (3-D) or in-plane images (2-D, isolated along z):
  ``u(R) = 2 [erf(sqrt(b) R) - erf(sqrt(c) R)]/R``;
* the plain Gygi-Baldereschi element misses only the curvature of the pair density at ``Q = 0``:
  ``E - E_isolated = 4 pi/(a Omega N_k)`` Ry for the Gaussian (``|g(Q)|^2 = exp(-Q**2/(2 a))``).
"""

from __future__ import annotations

import itertools
import unittest

import numpy as np
from scipy.special import erf

from parsec_python.Hartree.exchange_kernel import PeriodicExchangeKernel

FCC = 0.5 * 10.26 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
OMEGA = 0.11


def self_energies(exponent: float, omega: float = OMEGA) -> dict:
    """Isolated self-energies (Ry) of a unit Gaussian for the three kernels."""

    b = exponent / 2.0
    c = 1.0 / (1.0 / b + 1.0 / omega**2)
    coulomb = 4.0 * np.sqrt(b / np.pi)
    long_range = 4.0 * np.sqrt(c / np.pi)
    return {"coulomb": coulomb, "erf": long_range, "erfc": coulomb - long_range}


def pair_energy(exponent: float, separation: float, omega: float = OMEGA) -> float:
    """erfc interaction energy (Ry) of two unit Gaussians ``separation`` apart."""

    if separation == 0.0:
        return self_energies(exponent, omega)["erfc"]
    b = exponent / 2.0
    c = 1.0 / (1.0 / b + 1.0 / omega**2)
    return 2.0 * (erf(np.sqrt(b) * separation) - erf(np.sqrt(c) * separation)) / separation


def mesh_vectors(kernel: PeriodicExchangeKernel, mesh) -> list:
    """The Cartesian q of the Gamma-centred mesh (periodic axes of the kernel)."""

    reciprocal = kernel.reciprocal
    return [
        sum(index[axis] / mesh[axis] * reciprocal[axis] for axis in range(3))
        for index in itertools.product(*(range(count) for count in mesh))
    ]


def bloch_gaussian(lattice, shape, q, exponent, center, periodic=3, images=3) -> np.ndarray:
    """Periodic part ``sum_R g(r - R) exp(-i q.(r - R))`` of a Bloch sum of Gaussians on the grid box."""

    counts = np.asarray(shape)
    index = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), -1).reshape(-1, 3)
    points = (index / counts) @ lattice
    values = np.zeros(len(points), dtype=complex)
    for translation in itertools.product(range(-images, images + 1), repeat=periodic):
        shift = np.array(list(translation) + [0] * (3 - periodic), dtype=float) @ lattice
        offset = points - center - shift
        values += (
            (exponent / np.pi) ** 1.5
            * np.exp(-exponent * np.einsum("ij,ij->i", offset, offset))
            * np.exp(-1j * (offset @ q))
        )
    return values.reshape(shape)


def grid_energy(kernel, lattice, mesh, exponent, center, periodic=3) -> float:
    """``(1/N_k) sum_q dV sum_r conj(p_q) w_q`` through :meth:`PeriodicExchangeKernel.potential` (the contract)."""

    volume_element = abs(np.linalg.det(lattice)) / np.prod(kernel.box_shape)
    total = 0.0
    for q in mesh_vectors(kernel, mesh):
        density = bloch_gaussian(lattice, kernel.box_shape, q, exponent, center, periodic)
        total += volume_element * float(np.vdot(density, kernel.potential(density, q)).real)
    return total / int(np.prod(mesh))


def spectral_energy(kernel, mesh, structure) -> float:
    """``(1/N_k) sum_q (1/Omega) sum_G values(q)[G] S(q + G)`` for a structure factor ``S`` of the FFT grid."""

    frequencies = np.stack(
        np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in kernel.shape), indexing="ij"), -1
    )
    vectors = frequencies @ kernel.reciprocal
    total = 0.0
    for q in mesh_vectors(kernel, mesh):
        total += float(np.sum(kernel.values(q) * structure(q + vectors))) / kernel.volume
    return total / int(np.prod(mesh))


def gaussian_structure(exponent: float):
    return lambda vectors: np.exp(-np.sum(vectors * vectors, axis=-1) / (2.0 * exponent))


class BulkKernelTests(unittest.TestCase):
    def test_grid_contract_matches_the_erfc_supercell_sum(self) -> None:
        # erfc with its exact limit at q + G = 0: the Born-von Karman supercell of the mesh, summed in real space.
        exponent = 1.0
        mesh = (2, 2, 2)
        kernel = PeriodicExchangeKernel(
            FCC, (18, 18, 18), mesh, alpha=0.0, beta=1.0, omega=OMEGA, divergence="none"
        )
        energy = grid_energy(kernel, FCC, mesh, exponent, np.zeros(3))
        supercell = np.asarray(mesh)[:, None] * FCC
        reference = 0.0
        for translation in itertools.product(range(-12, 13), repeat=3):
            separation = float(np.linalg.norm(np.asarray(translation) @ supercell))
            if separation < 120.0:
                reference += pair_energy(exponent, separation)
        self.assertAlmostEqual(energy, reference, delta=1.0e-9)

    def test_plain_gygi_baldereschi_misses_only_the_curvature(self) -> None:
        exponent = 1.5
        isolated = self_energies(exponent)["coulomb"]
        volume = abs(np.linalg.det(FCC))
        for count in (1, 2, 3):
            mesh = (count,) * 3
            kernel = PeriodicExchangeKernel(FCC, (16, 16, 16), mesh, alpha=1.0, beta=0.0, x_gamma_extrapolation=False)
            energy = spectral_energy(kernel, mesh, gaussian_structure(exponent))
            curvature = 4.0 * np.pi / (exponent * volume * count**3)
            self.assertAlmostEqual(energy - isolated, curvature, delta=1.0e-7)

    def test_default_treatment_converges_for_every_kernel(self) -> None:
        exponent = 1.5
        isolated = self_energies(exponent)
        mesh = (4, 4, 4)
        for name, (alpha, beta) in {"coulomb": (1.0, 0.0), "erfc": (0.0, 1.0), "erf": (1.0, -1.0)}.items():
            # erf with omega = 0.11 on this mesh (spacing 2.4 omega) is where the automatic choice guards the sign
            # of the kernel (test_negative_erfc_coefficient_keeps_the_operator_definite); "coulomb" by name here.
            kernel = PeriodicExchangeKernel(
                FCC, (16, 16, 16), mesh, alpha=alpha, beta=beta, omega=OMEGA if beta else None,
                x_gamma_extrapolation="coulomb" if beta < 0.0 else None,
            )
            self.assertEqual(kernel.x_gamma_extrapolation, "coulomb")
            self.assertEqual(kernel.auxiliary_exponent, 0.0)
            energy = spectral_energy(kernel, mesh, gaussian_structure(exponent))
            self.assertLess(abs(energy - isolated[name]), 1.0e-5, name)
        # Quantum ESPRESSO's default (extrapolation on every component) is far less accurate for erf here.
        kernel = PeriodicExchangeKernel(
            FCC, (16, 16, 16), mesh, alpha=1.0, beta=-1.0, omega=OMEGA, x_gamma_extrapolation=True
        )
        energy = spectral_energy(kernel, mesh, gaussian_structure(exponent))
        self.assertGreater(abs(energy - isolated["erf"]), 1.0e-4)

    def test_kernel_is_linear_in_the_cam_coefficients(self) -> None:
        q = mesh_vectors(PeriodicExchangeKernel(FCC, (8, 8, 8), (2, 2, 2), alpha=1.0, beta=0.0), (2, 2, 2))
        for extrapolation in (False, True, "coulomb"):
            parts = [
                PeriodicExchangeKernel(
                    FCC, (8, 8, 8), (2, 2, 2), alpha=alpha, beta=beta, omega=OMEGA if beta else None,
                    x_gamma_extrapolation=extrapolation,
                )
                for alpha, beta in ((1.0, 0.0), (0.0, 1.0), (0.2, 0.55))
            ]
            for vector in (q[0], q[3]):
                np.testing.assert_allclose(
                    parts[2].values(vector),
                    0.2 * parts[0].values(vector) + 0.55 * parts[1].values(vector),
                    rtol=1.0e-13,
                    atol=1.0e-12,
                )
            self.assertAlmostEqual(
                parts[2].divergence_element,
                0.2 * parts[0].divergence_element + 0.55 * parts[1].divergence_element,
                delta=1.0e-9 * abs(parts[0].divergence_element),
            )

    def test_long_range_kernel_has_no_cancellation(self) -> None:
        kernel = PeriodicExchangeKernel(FCC, (12, 12, 12), (1, 1, 1), alpha=0.3, beta=-0.3, omega=0.4,
                                        x_gamma_extrapolation=False)
        values = kernel.values(np.zeros(3))
        frequencies = np.stack(np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in kernel.shape), indexing="ij"), -1)
        squared = np.sum((frequencies @ kernel.reciprocal) ** 2, axis=-1)
        squared[0, 0, 0] = 1.0
        expected = 0.3 * 8.0 * np.pi * np.exp(-squared / (4.0 * 0.4**2)) / squared
        inside = np.ones(kernel.shape, dtype=bool)
        inside[0, 0, 0] = False
        for axis in range(3):  # the Nyquist planes hold alias means
            index = [slice(None)] * 3
            index[axis] = kernel.shape[axis] // 2
            inside[tuple(index)] = False
        np.testing.assert_allclose(values[inside], expected[inside], rtol=1.0e-13, atol=1.0e-300)

    def test_time_reversal_and_reality_on_an_oblique_cell(self) -> None:
        kernel = PeriodicExchangeKernel(FCC, (10, 12, 9), (4, 4, 4), alpha=0.25, beta=0.0)
        reverse = [(-np.arange(n)) % n for n in kernel.shape]
        for q in mesh_vectors(kernel, (4, 4, 4))[1:20:3]:
            forward = kernel.values(q)
            backward = kernel.values(-q)
            np.testing.assert_allclose(backward[np.ix_(*reverse)], forward, rtol=1.0e-15)
        density = np.random.default_rng(5).normal(size=kernel.box_shape)
        potential = np.fft.ifftn(kernel.values(np.zeros(3)) * np.fft.fftn(density))
        self.assertLess(np.max(np.abs(potential.imag)), 1.0e-14 * np.max(np.abs(potential.real)))

    def test_singular_element_follows_q(self) -> None:
        kernel = PeriodicExchangeKernel(FCC, (9, 9, 9), (2, 2, 2), alpha=1.0, beta=0.0)
        at_zero = kernel.values(np.zeros(3))
        self.assertEqual(at_zero[0, 0, 0], kernel.divergence_element)
        q = kernel.reciprocal[0] - kernel.reciprocal[2]
        self.assertTrue(kernel.is_singular(q))
        shifted = kernel.values(q)
        self.assertEqual(shifted[-1, 0, 1], kernel.divergence_element)  # G = -q
        # values(q)[m] = K(q + G_m) = values(0)[m + (1, 0, -1)] while both frequencies lie in -4..4.
        for m in itertools.product(range(-3, 4), repeat=3):
            target = (m[0] + 1, m[1], m[2] - 1)
            if max(abs(value) for value in target) <= 4:
                self.assertAlmostEqual(
                    shifted[m[0] % 9, m[1] % 9, m[2] % 9],
                    at_zero[target[0] % 9, target[1] % 9, target[2] % 9],
                    delta=1.0e-13 * abs(kernel.divergence_element),
                )

    def test_element_matches_quantum_espresso(self) -> None:
        # Quantum ESPRESSO 7.5's kernel element at q + G = 0 (Ry, the units of 8 pi/Q^2) for Si (a = 10.26 bohr,
        # ecutwfc 60 Ry, so alpha' = 10/60 bohr^2): -exxdiv of exx_divergence (+ 2 pi/w^2 for erfc without
        # x_gamma), recomputed from its formula by the reference stream of the round (refs/pbc/qe/exxdiv.py).
        reference = {
            ((1, 1, 1), "pbe0", False): 241.3187134365,
            ((1, 1, 1), "pbe0", True): 206.8446114322,
            ((2, 2, 2), "pbe0", False): 965.2748537460,
            ((2, 2, 2), "pbe0", True): 827.3784460680,
            ((3, 3, 3), "pbe0", True): 1861.6015036529,
            ((1, 1, 1), "hse06", False): 170.3700960312,
            ((2, 2, 2), "hse06", False): 429.2121202245,
            ((2, 2, 2), "hse06", True): 295.8194562209,
            ((3, 3, 3), "hse06", False): 513.2720453597,
        }
        for (mesh, functional, extrapolation), element in reference.items():
            coefficients = dict(alpha=0.0, beta=1.0, omega=OMEGA) if functional == "hse06" else dict(alpha=1.0, beta=0.0)
            kernel = PeriodicExchangeKernel(
                FCC, (8, 8, 8), mesh, x_gamma_extrapolation=extrapolation, auxiliary_exponent=10.0 / 60.0,
                **coefficients,
            )
            self.assertAlmostEqual(kernel.divergence_element, element, delta=1.0e-12 * element)
        none = PeriodicExchangeKernel(FCC, (8, 8, 8), (2, 2, 2), alpha=0.0, beta=1.0, omega=OMEGA, divergence="none")
        self.assertAlmostEqual(none.divergence_element, 2.0 * np.pi / OMEGA**2, delta=1.0e-10)

    def test_q_is_snapped_onto_the_mesh(self) -> None:
        # A q within the tolerance of a mesh point (k-points given to 8 digits) is that mesh point exactly.
        for extrapolation in (False, True, "coulomb"):
            kernel = PeriodicExchangeKernel(
                FCC, (12, 12, 12), (2, 2, 2), alpha=0.25, beta=0.0, x_gamma_extrapolation=extrapolation
            )
            near_zero = 2.0e-8 * kernel.reciprocal[0]
            near_half = kernel.reciprocal[1] / 2.0 + 3.0e-8 * kernel.reciprocal[2]
            np.testing.assert_array_equal(kernel.values(near_zero), kernel.values(np.zeros(3)))
            np.testing.assert_array_equal(kernel.values(near_half), kernel.values(kernel.reciprocal[1] / 2.0))
            self.assertTrue(kernel.is_singular(near_zero))
            self.assertEqual(kernel.values(near_zero)[0, 0, 0], kernel.divergence_element)

    def test_potential_of_a_real_density_is_real_only_at_q_zero(self) -> None:
        kernel = PeriodicExchangeKernel(FCC, (10, 10, 10), (2, 2, 2), alpha=1.0, beta=0.0)
        density = np.random.default_rng(2).normal(size=kernel.box_shape)
        self.assertFalse(np.iscomplexobj(kernel.potential(density, np.zeros(3))))
        shifted = kernel.potential(density, kernel.reciprocal[0])  # q = b_1: K(b_1 + G) != K(b_1 - G)
        self.assertTrue(np.iscomplexobj(shifted))
        self.assertGreater(np.max(np.abs(shifted.imag)), 0.1 * np.max(np.abs(shifted.real)))

    def test_closed_form_elements_equal_the_reciprocal_sums(self) -> None:
        # Ewald's identity against the direct sum over the reciprocal lattice of the supercell (moderate a).
        for extrapolation in (False, True, "coulomb"):
            for coefficients in (
                dict(alpha=1.0, beta=0.0),
                dict(alpha=0.0, beta=1.0, omega=OMEGA),
                dict(alpha=1.0, beta=-1.0, omega=0.3),
            ):
                kernel = PeriodicExchangeKernel(
                    FCC, (8, 8, 8), (3, 2, 4), x_gamma_extrapolation=extrapolation, auxiliary_exponent=0.4,
                    **coefficients,
                )
                direct = 2.0 * sum(
                    coefficient
                    * (
                        (0.0 if weighted else kernel._origin_value(name))
                        + kernel.k_points * (kernel._auxiliary_integral(name) - kernel._auxiliary_sum(name, weighted))
                    )
                    for name, coefficient, weighted in kernel._components()
                )
                self.assertAlmostEqual(kernel.divergence_element, direct, delta=1.0e-12 * abs(direct))

    def test_coulomb_part_does_not_depend_on_the_auxiliary_exponent(self) -> None:
        exponent = 1.5
        energies = []
        for auxiliary in (0.0, 0.05, 0.2):
            kernel = PeriodicExchangeKernel(
                FCC, (16, 16, 16), (2, 2, 2), alpha=1.0, beta=0.0, auxiliary_exponent=auxiliary
            )
            energies.append(spectral_energy(kernel, (2, 2, 2), gaussian_structure(exponent)))
        self.assertLess(max(energies) - min(energies), 1.0e-10)

    def test_constructor_cost_does_not_grow_with_cell_and_mesh(self) -> None:
        import time

        started = time.perf_counter()
        PeriodicExchangeKernel(np.eye(3) * 20.5, (128, 128, 128), (4, 4, 4), alpha=0.19, beta=0.46, omega=0.33)
        PeriodicExchangeKernel(np.eye(3) * 20.5, (128, 128, 128), (8, 8, 8), alpha=0.0, beta=0.25, omega=0.11)
        self.assertLess(time.perf_counter() - started, 2.0)

    def test_negative_erfc_coefficient_keeps_the_operator_definite(self) -> None:
        # -U^H V_x U of occupied and empty bands must be positive (ACE's Cholesky); for beta < 0 the automatic
        # choice weights both components unless the mesh resolves omega (spacing <= 1.5 omega).
        omega = 0.3
        coarse = PeriodicExchangeKernel(FCC, (12, 12, 12), (1, 1, 1), alpha=1.0, beta=-1.0, omega=omega)
        fine = PeriodicExchangeKernel(FCC, (12, 12, 12), (3, 3, 3), alpha=1.0, beta=-1.0, omega=omega)
        self.assertIs(coarse.x_gamma_extrapolation, True)  # spacing 1.06 = 3.5 omega
        self.assertEqual(fine.x_gamma_extrapolation, "coulomb")  # spacing 0.35 = 1.2 omega
        # The sign test with omega = 0.11 (spacing 9.6 omega at Gamma), where the forced weights fail clearly.
        coarse = PeriodicExchangeKernel(FCC, (16, 16, 16), (1, 1, 1), alpha=1.0, beta=-1.0, omega=OMEGA)
        self.assertIs(coarse.x_gamma_extrapolation, True)
        forced = PeriodicExchangeKernel(
            FCC, (16, 16, 16), (1, 1, 1), alpha=1.0, beta=-1.0, omega=OMEGA, x_gamma_extrapolation="coulomb"
        )
        rng = np.random.default_rng(1)
        shape = coarse.box_shape
        frequencies = np.stack(np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in shape), indexing="ij"), -1)
        squared = np.sum((frequencies @ coarse.reciprocal) ** 2, axis=-1)
        occupied, empty = 4, 6
        bands = occupied + empty
        smallest = {"automatic": np.inf, "forced": np.inf}
        for _trial in range(20):
            width = rng.uniform(0.7, 4.0)
            coefficients = rng.normal(size=(bands,) + shape) + 1j * rng.normal(size=(bands,) + shape)
            orbitals = np.fft.ifftn(coefficients * np.exp(-squared / (2.0 * width**2)), axes=(1, 2, 3))
            orbitals = np.linalg.qr(orbitals.reshape(bands, -1).T)[0].T
            for label, kernel in (("automatic", coarse), ("forced", forced)):
                values = kernel.values(np.zeros(3)).reshape(-1)
                matrix = np.zeros((bands, bands), dtype=complex)
                for m in range(occupied):
                    pairs = (np.conj(orbitals[m])[None] * orbitals).reshape((-1,) + shape)
                    transformed = np.fft.fftn(pairs, axes=(1, 2, 3)).reshape(bands, -1)
                    matrix += (np.conj(transformed) * values[None]) @ transformed.T
                eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
                smallest[label] = min(smallest[label], eigenvalues.min() / eigenvalues.max())
        self.assertGreater(smallest["automatic"], 0.0)
        self.assertLess(smallest["forced"], 0.0)  # what the guard prevents

    def test_refusals(self) -> None:
        kernel = PeriodicExchangeKernel(FCC, (8, 8, 8), (2, 2, 2), alpha=1.0, beta=0.0)
        with self.assertRaises(ValueError):
            kernel.values(kernel.reciprocal[0] / 3.0)  # not a difference of mesh points
        with self.assertRaises(ValueError):
            PeriodicExchangeKernel(FCC, (8, 8, 8), (2, 2, 2), alpha=0.0, beta=1.0)  # no omega
        with self.assertRaises(ValueError):
            PeriodicExchangeKernel(FCC, (8, 8, 8), (2, 2, 2), alpha=1.0, beta=0.0, periodic_axes=(0,))
        with self.assertRaises(ValueError):
            PeriodicExchangeKernel(FCC, (8, 8, 8), (2, 2, 2), alpha=1.0, beta=0.0, divergence="vcut")
        with self.assertRaises(ValueError):
            PeriodicExchangeKernel(FCC, (8, 8, 8), (2, 2, 2), alpha=1.0, beta=0.0, x_gamma_extrapolation="erfc")


SLAB_SIDE = 6.0
SLAB_STEP = 0.5


def slab_lattice(planes: int) -> np.ndarray:
    return np.array([[SLAB_SIDE, 0.0, 0.0], [0.0, SLAB_SIDE, 0.0], [0.0, 0.0, planes * SLAB_STEP]])


def anisotropic_self_energy(exponents, omega=None) -> float:
    """Isolated self-energy (Ry) of ``1/r`` (or ``erf(w r)/r``) for ``S(Q) = exp(-sum_i Q_i^2/(2 e_i))``.

    ``(1/(2 pi)^3) int 8 pi S(Q)/Q^2 d^3Q`` with ``1/Q^2 = int_0^inf exp(-t Q^2) dt``: a one-dimensional integral of
    ``pi^1.5/sqrt(prod_i (d_i + t))``, ``d_i = 1/(2 e_i)`` (plus ``1/(4 w^2)`` for erf).
    """

    from scipy.integrate import quad

    widths = 1.0 / (2.0 * np.asarray(exponents, dtype=float))
    if omega is not None:
        widths = widths + 1.0 / (4.0 * omega**2)
    value, _error = quad(lambda t: np.pi**1.5 / np.sqrt(np.prod(widths + t)), 0.0, np.inf, epsabs=0.0, epsrel=1e-13)
    return 8.0 * np.pi * value / (2.0 * np.pi) ** 3


class SlabKernelTests(unittest.TestCase):
    def test_truncation_gives_the_isolated_slab_for_any_box_height(self) -> None:
        exponent = 0.5
        mesh = (2, 2, 1)
        reference = 0.0
        for translation in itertools.product(range(-70, 71), repeat=2):
            separation = SLAB_SIDE * mesh[0] * float(np.hypot(*translation))
            if separation < 400.0:
                reference += pair_energy(exponent, separation)
        for planes in (32, 40):
            lattice = slab_lattice(planes)
            kernel = PeriodicExchangeKernel(
                lattice, (12, 12, planes), mesh, alpha=0.0, beta=1.0, omega=OMEGA,
                periodic_axes=(0, 1), divergence="none", slab_coulomb="truncated",
            )
            self.assertEqual(kernel.shape, (12, 12, 2 * planes))
            center = np.array([0.0, 0.0, 0.5 * planes * SLAB_STEP])
            energy = grid_energy(kernel, lattice, mesh, exponent, center, periodic=2)
            self.assertAlmostEqual(energy, reference, delta=1.0e-11)

    def test_truncated_treatment_converges(self) -> None:
        exponent = 0.5
        isolated = self_energies(exponent)
        planes = 32
        lattice = slab_lattice(planes)
        # A Gaussian well inside the padded box: |c(Q)|^2 = exp(-Q^2/(2a))/Omega^2 wherever it sits.
        for name, (alpha, beta), bound in (("coulomb", (1.0, 0.0), 1.0e-10), ("erf", (1.0, -1.0), 1.0e-5)):
            mesh = (4, 4, 1)
            kernel = PeriodicExchangeKernel(
                lattice, (12, 12, planes), mesh, alpha=alpha, beta=beta, omega=OMEGA if beta else None,
                periodic_axes=(0, 1), slab_coulomb="truncated",
            )
            self.assertIs(kernel.x_gamma_extrapolation, False)
            energy = spectral_energy(kernel, mesh, gaussian_structure(exponent))
            self.assertLess(abs(energy - isolated[name]), bound, name)

    def test_padded_default_handles_anisotropic_pair_densities(self) -> None:
        # The default slab kernel is the bulk kernel on the box of twice the height; for a pair density of a slab
        # (narrower along z than in the plane, or the reverse) it converges far faster than the truncated one.
        planes = 32
        lattice = slab_lattice(planes)
        exponents = np.array([1.0, 0.6, 0.35])

        def structure(vectors):
            return np.exp(-np.sum(vectors * vectors / (2.0 * exponents), axis=-1))

        mesh = (4, 4, 1)
        for name, (alpha, beta) in (("coulomb", (1.0, 0.0)), ("erf", (1.0, -1.0))):
            isolated = anisotropic_self_energy(exponents, OMEGA if beta else None)
            coefficients = dict(alpha=alpha, beta=beta, omega=OMEGA if beta else None, periodic_axes=(0, 1))
            padded = PeriodicExchangeKernel(
                lattice, (12, 12, planes), mesh, x_gamma_extrapolation="coulomb", **coefficients
            )
            truncated = PeriodicExchangeKernel(lattice, (12, 12, planes), mesh, slab_coulomb="truncated", **coefficients)
            self.assertEqual(padded.slab_coulomb, "padded")
            default = PeriodicExchangeKernel(lattice, (12, 12, planes), mesh, alpha=1.0, beta=0.0, periodic_axes=(0, 1))
            self.assertEqual(default.x_gamma_extrapolation, "coulomb")
            bulk = PeriodicExchangeKernel(
                np.diag([SLAB_SIDE, SLAB_SIDE, 2 * planes * SLAB_STEP]), (12, 12, 2 * planes), mesh,
                alpha=alpha, beta=beta, omega=OMEGA if beta else None, x_gamma_extrapolation="coulomb",
            )
            q = padded.reciprocal[0] / 4 + 3 * padded.reciprocal[1] / 4
            np.testing.assert_array_equal(padded.values(q), bulk.values(q))
            padded_error = spectral_energy(padded, mesh, structure) - isolated
            truncated_error = spectral_energy(truncated, mesh, structure) - isolated
            # erf: its erfc part is not resolved by this mesh (spacing 2.4 omega) and keeps an a-dependent 3e-5
            self.assertLess(abs(padded_error), 2.0e-5 if beta == 0.0 else 5.0e-5, name)
            self.assertGreater(abs(truncated_error), 5.0e-4, name)

    def test_slab_refusals(self) -> None:
        lattice = slab_lattice(16)
        with self.assertRaises(ValueError):
            PeriodicExchangeKernel(lattice, (8, 8, 16), (2, 2, 2), alpha=1.0, beta=0.0, periodic_axes=(0, 1))
        kernel = PeriodicExchangeKernel(lattice, (8, 8, 16), (2, 2, 1), alpha=1.0, beta=0.0, periodic_axes=(0, 1))
        with self.assertRaises(ValueError):
            kernel.values(np.array([0.0, 0.0, 0.1]))
        tilted = lattice.copy()
        tilted[2, 0] = 1.0
        with self.assertRaises(ValueError):
            PeriodicExchangeKernel(tilted, (8, 8, 16), (2, 2, 1), alpha=1.0, beta=0.0, periodic_axes=(0, 1))
        with self.assertRaises(ValueError):
            PeriodicExchangeKernel(lattice, (8, 8, 16), (2, 2, 1), alpha=1.0, beta=0.0, periodic_axes=(0, 1),
                                   slab_coulomb="periodic")
        with self.assertRaises(ValueError):
            PeriodicExchangeKernel(lattice, (8, 8, 16), (2, 2, 1), alpha=1.0, beta=0.0, slab_coulomb="truncated")


if __name__ == "__main__":
    unittest.main()
