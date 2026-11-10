"""The exchange of periodic hybrids on the devices (``acceleration/Hamiltonian/bloch_exchange_device.py``).

Every device piece is compared with a plain NumPy evaluation of the formulas
of :mod:`parsec_python.V_xc.bloch_exchange` written here pair by pair:

``W_t[:, n] = -(1/N) sum_(k' in mesh) sum_(m occ) f_mk' u_mk' ifftn(K(k_t - k' + G) fftn(conj(u_mk') u_nt / dV))``,
``E_x = (s/2) sum_t w_t sum_n f_nt Re <u_nt|W_t[:, n]>``, ``M = -U^H W = L L^H``, ``xi = W L^-H``,

on small oblique cells with random orthonormal Bloch factors and a test
kernel ``8 pi / |q+G|^2`` (``q + G = 0`` -> 0, the Nyquist planes averaged).
The device results agree with it to round-off (1e-12 relative).
"""

from __future__ import annotations

import itertools
import unittest
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.backends.cupy import cupy_available


class _TestKernel:
    """``8 pi/|q+G|^2`` on an FFT box of the lattice ``box_lattice`` (rows), Nyquist components averaged."""

    def __init__(self, box_lattice, shape, *, screening: float | None = None) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(box_lattice, dtype=float)).T
        self.screening = screening
        self.calls = 0

    def values(self, q, xp=np):
        self.calls += 1
        q = np.asarray(q, dtype=float).reshape(3)
        integers = [np.fft.fftfreq(n, d=1.0 / n) for n in self.shape]
        choices = []
        for axis, n in enumerate(self.shape):
            options = [integers[axis]]
            if n % 2 == 0:
                flipped = integers[axis].copy()
                flipped[n // 2] = n // 2
                options.append(flipped)
            choices.append(options)
        total = np.zeros(self.shape)
        combinations = list(itertools.product(*choices))
        for m0, m1, m2 in combinations:
            g = (
                m0[:, None, None, None] * self.reciprocal[0]
                + m1[None, :, None, None] * self.reciprocal[1]
                + m2[None, None, :, None] * self.reciprocal[2]
            ) + q
            squared = np.sum(g * g, axis=-1)
            values = np.zeros_like(squared)
            nonzero = squared > 1.0e-12
            values[nonzero] = 8.0 * np.pi / squared[nonzero]
            if self.screening is not None:
                values[nonzero] *= 1.0 - np.exp(-squared[nonzero] / (4.0 * self.screening**2))
                values[~nonzero] = 2.0 * np.pi / self.screening**2
            total += values
        total /= len(combinations)
        return xp.asarray(total)


def _orthonormal(rng, points: int, states: int, complex_values: bool = True) -> np.ndarray:
    block = rng.standard_normal((points, states))
    if complex_values:
        block = block + 1j * rng.standard_normal((points, states))
    q, _ = np.linalg.qr(block)
    return q


class _Case:
    """A small cell: box, lattice, k-mesh representatives, orbitals and occupations."""

    def __init__(self, *, box_shape=(6, 5, 7), fft_shape=None, flat=None, mesh=(3, 1, 1), states=5,
                 seed=7, screening=None) -> None:
        rng = np.random.default_rng(seed)
        self.lattice = np.array([[5.0, 0.3, 0.0], [0.8, 4.6, 0.2], [0.0, 0.4, 6.1]])
        self.box_shape = tuple(box_shape)
        self.fft_shape = tuple(fft_shape or box_shape)
        self.flat = flat
        points = int(np.prod(self.box_shape)) if flat is None else int(np.asarray(flat).size)
        self.points = points
        self.volume_element = abs(np.linalg.det(self.lattice)) / float(np.prod(self.box_shape))
        stretch = np.array(self.fft_shape, dtype=float) / np.array(self.box_shape, dtype=float)
        self.kernel = _TestKernel(self.lattice * stretch[:, None], self.fft_shape, screening=screening)
        reciprocal = 2.0 * np.pi * np.linalg.inv(self.lattice).T
        # Representatives of the mesh under time reversal: k ~ -k mod G.
        fractions = [
            np.array([i / mesh[0], j / mesh[1], l / mesh[2]])
            for i in range(mesh[0])
            for j in range(mesh[1])
            for l in range(mesh[2])
        ]
        fractions = [f - np.round(f) for f in fractions]

        def same(a, b):
            difference = a - b
            return bool(np.allclose(difference - np.round(difference), 0.0))

        representatives = []
        for f in fractions:
            if not any(same(f, other) or same(f, -other) for other in representatives):
                representatives.append(f)
        self.mesh_size = len(fractions)
        # -k is another mesh point unless 2k is a reciprocal-lattice vector
        self.partners = np.array([not same(f, -f) for f in representatives], dtype=bool)
        self.k_points = np.array([f @ reciprocal for f in representatives])
        self.weights = np.array([(2.0 if p else 1.0) / self.mesh_size for p in self.partners])
        assert int(len(representatives) + np.count_nonzero(self.partners)) == self.mesh_size
        self.orbitals = [_orthonormal(rng, points, states) for _ in representatives]
        base = np.array([1.0, 1.0, 0.7, 0.0, 0.0][:states] + [0.0] * max(0, states - 5))
        self.occupations = [np.clip(base - 0.05 * index, 0.0, 1.0) for index in range(len(representatives))]
        self.spin_weight = 2.0

    def to_box(self, values):
        box = np.zeros(int(np.prod(self.fft_shape)), dtype=complex)
        if self.flat is None:
            indices = np.arange(self.points)
        else:
            indices = np.asarray(self.flat)
        coordinates = np.unravel_index(indices, self.box_shape)
        box[np.ravel_multi_index(coordinates, self.fft_shape)] = values
        return box.reshape(self.fft_shape), np.ravel_multi_index(coordinates, self.fft_shape)

    def orbital_set(self, orbitals=None):
        return SimpleNamespace(
            k_points=self.k_points,
            weights=self.weights,
            orbitals=tuple(self.orbitals if orbitals is None else orbitals),
            occupations=tuple(self.occupations),
            spin_weight=self.spin_weight,
            partners=self.partners,
            mesh_size=self.mesh_size,
        )

    def box(self):
        return SimpleNamespace(shape=self.box_shape, flat=self.flat)

    # ----- the NumPy reference --------------------------------------------------------------

    def sources(self):
        found = []
        for rep, k in enumerate(self.k_points):
            occupied = np.flatnonzero(self.occupations[rep] > 1.0e-12)
            columns = self.orbitals[rep][:, occupied]
            weights = self.occupations[rep][occupied]
            found.append((k, columns, weights))
            if self.partners[rep]:
                found.append((-k, columns.conj(), weights))
        return found

    def reference_applied(self):
        applied = []
        for rep, k_target in enumerate(self.k_points):
            target = self.orbitals[rep]
            result = np.zeros_like(target)
            for k_source, columns, weights in self.sources():
                values = self.kernel.values(k_target - k_source)
                for n in range(target.shape[1]):
                    for m in range(columns.shape[1]):
                        rho = np.conj(columns[:, m]) * target[:, n] / self.volume_element
                        box, index = self.to_box(rho)
                        potential = np.fft.ifftn(values * np.fft.fftn(box)).reshape(-1)[index]
                        result[:, n] -= (weights[m] / self.mesh_size) * columns[:, m] * potential
            applied.append(result)
        return applied

    def reference_energy(self, applied):
        total = 0.0
        for rep in range(len(self.orbitals)):
            values = np.sum(np.conj(self.orbitals[rep]) * applied[rep], axis=0).real
            total += self.weights[rep] * float(values @ self.occupations[rep])
        return 0.5 * self.spin_weight * total


@unittest.skipUnless(cupy_available(), "CuPy and a CUDA device are required")
class BlochExchangeDeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        import cupy as cp

        self.cp = cp

    def _channel(self, case, devices=(0,)):
        from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import device_bloch_channel

        cp = self.cp
        orbitals = []
        for index, vectors in enumerate(case.orbitals):
            with cp.cuda.Device(devices[index % len(devices)]):
                # a strided view, as the eigensolver keeps (grid, working) and hands [:, :states]
                working = cp.zeros((vectors.shape[0], vectors.shape[1] + 3), dtype=cp.complex128)
                working[:, : vectors.shape[1]] = cp.asarray(vectors)
                orbitals.append(working[:, : vectors.shape[1]])
        return device_bloch_channel(case.orbital_set(orbitals), case.volume_element)

    def _solvers(self, case, batch_size=4):
        from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import DevicePeriodicPairSolvers

        return DevicePeriodicPairSolvers(case.kernel, case.box(), batch_size)

    def _check_application(self, case, *, batch_size=4):
        from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import apply_device_bloch_exchange

        cp = self.cp
        channel = self._channel(case)
        application = apply_device_bloch_exchange(channel, self._solvers(case, batch_size), (0,))
        reference = case.reference_applied()
        scale = max(float(np.abs(values).max()) for values in reference)
        for rep, expected in enumerate(reference):
            self.assertLess(float(np.abs(cp.asnumpy(application.applied[rep]) - expected).max()), 1.0e-12 * scale)
        energy = case.reference_energy(reference)
        self.assertAlmostEqual(application.energy, energy, delta=1.0e-12 * abs(energy))
        self.assertLess(energy, 0.0)
        return application, reference

    def test_bulk_mesh_with_partners_matches_the_pairwise_reference(self) -> None:
        case = _Case(mesh=(3, 1, 1))
        self.assertTrue(bool(np.any(case.partners)) and not bool(np.all(case.partners)))
        application, _ = self._check_application(case)
        # every target column with every occupied column of the three mesh points
        occupied = sum(int(np.count_nonzero(values > 1e-12)) * (2 if p else 1)
                       for values, p in zip(case.occupations, case.partners))
        # every target column with every occupied column of the three mesh points, the pairs of two
        # occupied states once for both (PARSEC_CUPY_BLOCH_PAIR_SYMMETRY): 90 one way, 72 here
        self.assertEqual(sum(v.size for v in case.occupations) * occupied, 90)
        self.assertEqual(application.pair_solves, 72)

    def test_two_dimensional_mesh_and_batch_sizes(self) -> None:
        from parsec_python.acceleration.Hamiltonian import bloch_exchange_device as module

        case = _Case(mesh=(2, 3, 1), box_shape=(5, 6, 4), states=4)
        # chunks of the box size (the default) and of Exx_Batch_Size alone
        self._check_application(case, batch_size=1)
        chunk = module.CHUNK_BYTES
        module.CHUNK_BYTES = 0
        self.addCleanup(setattr, module, "CHUNK_BYTES", chunk)
        for batch_size in (1, 3, 64):
            self._check_application(case, batch_size=batch_size)

    def test_padded_slab_box_and_index_map(self) -> None:
        box_shape = (4, 5, 6)
        rng = np.random.default_rng(3)
        # a confined grid: the points of the box in another order, some left out
        flat = rng.permutation(int(np.prod(box_shape)))[:100]
        case = _Case(box_shape=box_shape, fft_shape=(4, 5, 12), flat=flat, mesh=(2, 2, 1), states=4)
        self._check_application(case)

    def test_screened_kernel(self) -> None:
        self._check_application(_Case(mesh=(3, 1, 1), screening=0.11))

    def test_occupied_columns_then_the_empty_ones(self) -> None:
        from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import apply_device_bloch_exchange

        cp = self.cp
        case = _Case(mesh=(3, 1, 1))
        channel = self._channel(case)
        solvers = self._solvers(case)
        occupied = apply_device_bloch_exchange(channel, solvers, (0,), columns="occupied")
        reference = case.reference_applied()
        self.assertAlmostEqual(occupied.energy, case.reference_energy(reference), delta=1e-12)
        complete = apply_device_bloch_exchange(channel, solvers, (0,), known=occupied)
        scale = max(float(np.abs(values).max()) for values in reference)
        for rep, expected in enumerate(reference):
            self.assertEqual(len(complete.columns[rep]), case.orbitals[rep].shape[1])
            self.assertLess(float(np.abs(cp.asnumpy(complete.applied[rep]) - expected).max()), 1e-12 * scale)

    def test_ace_reproduces_the_exchange_on_the_orbitals(self) -> None:
        from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import (
            apply_device_bloch_exchange,
            compress_device_bloch_exchange,
            device_cross_energy,
        )

        cp = self.cp
        case = _Case(mesh=(3, 1, 1))
        channel = self._channel(case)
        application = apply_device_bloch_exchange(channel, self._solvers(case), (0,))
        reference = case.reference_applied()
        operators = compress_device_bloch_exchange(application, fraction=0.25)
        cross = 0.0
        for rep, operator in enumerate(operators):
            self.assertLess(operator.asymmetry, 1e-12)
            vectors = cp.asarray(case.orbitals[rep])
            applied = cp.asnumpy(operator.apply(vectors, scaled=False))
            scale = float(np.abs(reference[rep]).max())
            self.assertLess(float(np.abs(applied - reference[rep]).max()), 1e-11 * scale)
            np.testing.assert_allclose(cp.asnumpy(operator.apply(vectors)), 0.25 * applied, atol=1e-13)
            # off the span, the compressed exchange is an upper bound of the exact one
            values = cp.asnumpy(operator.expectation_values(vectors, scaled=False))
            exact = np.sum(np.conj(case.orbitals[rep]) * reference[rep], axis=0).real
            np.testing.assert_allclose(values, exact, atol=1e-11 * scale)
            cross += float(cp.asnumpy(device_cross_energy(
                cp, operator, vectors, case.occupations[rep], case.weights[rep], case.spin_weight
            ))[0])
        # E_x(P0, P0) through the compressed operator is E_x(P0)
        self.assertAlmostEqual(cross, case.reference_energy(reference), delta=1e-11 * abs(cross))

    def test_ace_refuses_an_exchange_matrix_that_is_not_positive_definite(self) -> None:
        """As the host's Cholesky does (cuSOLVER's info flag is asked for, the factor's diagonal checked)."""

        from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import (
            _device_cholesky,
            apply_device_bloch_exchange,
            compress_device_bloch_exchange,
        )

        cp = self.cp
        with self.assertRaises(np.linalg.LinAlgError):
            _device_cholesky(cp, cp.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=cp.complex128))
        lower = cp.asnumpy(_device_cholesky(cp, cp.asarray([[4.0, 2.0], [2.0, 3.0]], dtype=cp.complex128)))
        np.testing.assert_allclose(lower @ lower.conj().T, [[4.0, 2.0], [2.0, 3.0]], atol=1e-14)
        case = _Case(mesh=(3, 1, 1))
        application = apply_device_bloch_exchange(self._channel(case), self._solvers(case), (0,))
        # an orbital whose W is zero: M = -U^H W has a zero row and column
        application.applied[0][:, 1] = 0.0
        with self.assertRaises(np.linalg.LinAlgError):
            compress_device_bloch_exchange(application)

    def test_device_bloch_operator_holds_the_exchange_term(self) -> None:
        from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import DeviceBlochACE
        from parsec_python.acceleration.Hamiltonian.device_bloch import (
            CupyArrayBackend,
            DeviceBlochConverter,
            NumpyArrayBackend,
        )

        rng = np.random.default_rng(5)
        points, rank, vectors = 60, 7, 5
        laplacian = sp.random(points, points, density=0.1, random_state=11, format="csr")
        laplacian = (laplacian + laplacian.T).tocsr()
        potential = rng.standard_normal(points)
        xi = rng.standard_normal((points, rank)) + 1j * rng.standard_normal((points, rank))
        block = rng.standard_normal((points, vectors)) + 1j * rng.standard_normal((points, vectors))
        base = SimpleNamespace(
            negative_laplacian=laplacian,
            _diagonal=potential,
            nonlocal_operator=SimpleNamespace(projectors=sp.csr_matrix((points, 0)), signs=np.zeros(0)),
        )
        expected = laplacian @ block + potential[:, None] * block - 0.3 * xi @ (xi.conj().T @ block)
        for backend in (NumpyArrayBackend(), CupyArrayBackend(0)):
            converter = DeviceBlochConverter(backend)
            for projectors in (xi, backend.asarray(np.asfortranarray(xi))):
                term = DeviceBlochACE(projectors=projectors, fraction=0.3)
                hamiltonian = SimpleNamespace(base=base, exchange=term)
                operator = converter.operator(hamiltonian)
                result = backend.asnumpy(operator.apply(backend.asarray(block)))
                np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12 * np.abs(expected).max())
            # without the term the base operator is unchanged
            plain = backend.asnumpy(converter.operator(base).apply(backend.asarray(block)))
            np.testing.assert_allclose(plain, laplacian @ block + potential[:, None] * block, atol=1e-12)
        released = DeviceBlochACE(projectors=xi, fraction=0.3)
        released.release()
        with self.assertRaises(RuntimeError):
            DeviceBlochConverter(NumpyArrayBackend()).operator(SimpleNamespace(base=base, exchange=released))

    def test_real_pair_solver_matches_the_complex_transform(self) -> None:
        """Gamma: the real-to-complex solve of the device loop of real orbitals."""

        cp = self.cp
        for fft_shape, flat in (((6, 5, 7), None), ((6, 4, 10), np.arange(6 * 4 * 5)[::-1].copy())):
            box_shape = (6, 5, 7) if flat is None else (6, 4, 5)
            case = _Case(box_shape=box_shape, fft_shape=fft_shape, flat=flat)
            solvers = self._solvers(case, batch_size=3)
            rng = np.random.default_rng(1)
            densities = rng.standard_normal((case.points, 3))
            potentials = cp.asnumpy(solvers.solve(cp, 0, cp.asfortranarray(cp.asarray(densities))))
            values = case.kernel.values(np.zeros(3))
            for column in range(3):
                box, index = case.to_box(densities[:, column])
                expected = np.fft.ifftn(values * np.fft.fftn(box)).reshape(-1)[index]
                self.assertLess(float(np.abs(expected.imag).max()), 1e-12 * float(np.abs(expected).max()))
                np.testing.assert_allclose(potentials[:, column], expected.real, rtol=0,
                                           atol=1e-12 * float(np.abs(expected).max()))

    def test_real_orbitals_through_the_device_loop_of_isolated_systems(self) -> None:
        """Gamma: apply_device_exchange with the periodic solver equals the real pairwise reference."""

        from parsec_python.acceleration.Hamiltonian.exchange_device import apply_device_exchange, channel_orbitals

        cp = self.cp
        case = _Case(box_shape=(6, 5, 7))
        rng = np.random.default_rng(2)
        orbitals = _orthonormal(rng, case.points, 6, complex_values=False)
        occupations = np.array([1.0, 1.0, 1.0, 0.4, 0.0, 0.0])
        channel = channel_orbitals(cp.asarray(orbitals), occupations, 2.0, case.volume_element)
        application = apply_device_exchange(channel, self._solvers(case, batch_size=3), (0,))
        values = case.kernel.values(np.zeros(3))
        expected = np.zeros_like(orbitals)
        for n in range(orbitals.shape[1]):
            for m in np.flatnonzero(occupations > 0):
                rho = orbitals[:, m] * orbitals[:, n] / case.volume_element
                potential = np.fft.ifftn(values * np.fft.fftn(rho.reshape(case.box_shape))).real.reshape(-1)
                expected[:, n] -= occupations[m] * orbitals[:, m] * potential
        np.testing.assert_allclose(cp.asnumpy(application.applied[0]), expected, rtol=0,
                                   atol=1e-12 * np.abs(expected).max())
        energy = float(np.sum(orbitals * expected, axis=0) @ occupations)  # (w/2) sum f <q|W>, w = 2
        self.assertAlmostEqual(application.energy, energy, delta=1e-12 * abs(energy))

    def test_kernel_cache_reuses_and_bounds_its_arrays(self) -> None:
        from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import DeviceKernelCache

        cp = self.cp
        case = _Case()
        cache = DeviceKernelCache(case.kernel, budget_bytes=2 * 8 * int(np.prod(case.fft_shape)))
        first = cache.values(cp, np.array([0.1, 0.0, 0.0]))
        again = cache.values(cp, np.array([0.1, 0.0, 0.0]) + 1e-15)
        self.assertIs(first, again)
        cache.values(cp, np.array([0.2, 0.0, 0.0]))
        cache.values(cp, np.array([0.3, 0.0, 0.0]))
        self.assertEqual(cache.device_bytes(cp), 2 * cache.array_bytes)
        self.assertEqual(cache.built, 3)

    def test_memory_rule_refuses_with_numbers(self) -> None:
        from parsec_python.acceleration.Hamiltonian import bloch_exchange_device as module

        case = _Case()
        channel = self._channel(case)
        solvers = self._solvers(case)
        planned = module.planned_bloch_exchange_bytes(channel, solvers, (0,))
        self.assertGreater(planned[0], 0)
        original = module.planned_bloch_exchange_bytes
        module.planned_bloch_exchange_bytes = lambda *args: {0: 1 << 60}
        try:
            with self.assertRaisesRegex(MemoryError, "GiB on device 0"):
                module.apply_device_bloch_exchange(channel, solvers, (0,))
        finally:
            module.planned_bloch_exchange_bytes = original

    def test_symmetric_pairs_equal_the_one_directional_sum(self) -> None:
        """One transform for both pairs of two occupied states (partners) gives every pair's sum."""

        import os

        from parsec_python.acceleration.Hamiltonian import bloch_exchange_device as module

        cp = self.cp
        # a 3x2x1 mesh: partnered and self-conjugate points; tiles of two columns cut the occupied lists
        case = _Case(mesh=(3, 2, 1), states=7)
        case.occupations = [np.array([1.0, 1.0, 0.9, 0.6, 0.3, 0.0, 0.0])] * len(case.orbitals)
        original = module.TILE_COLUMNS
        module.TILE_COLUMNS = 2
        try:
            results = {}
            for setting in ("off", "on"):
                os.environ[module.PAIR_SYMMETRY_SETTING] = setting
                channel = self._channel(case)
                application = module.apply_device_bloch_exchange(channel, self._solvers(case, batch_size=3), (0,))
                results[setting] = application
        finally:
            module.TILE_COLUMNS = original
            os.environ.pop(module.PAIR_SYMMETRY_SETTING, None)
        reference = case.reference_applied()
        scale = max(float(np.abs(values).max()) for values in reference)
        for setting, application in results.items():
            for rep, expected in enumerate(reference):
                self.assertLess(
                    float(np.abs(cp.asnumpy(application.applied[rep]) - expected).max()), 1.0e-12 * scale, setting
                )
        self.assertLess(results["on"].pair_solves, 0.75 * results["off"].pair_solves)


class DeviceBlochConverterCacheTests(unittest.TestCase):
    """The device copies of k.grad stay one per k-point however many SCF steps build new Hamiltonians."""

    def test_new_hamiltonians_of_one_k_point_share_one_device_k_gradient(self) -> None:
        from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter, NumpyArrayBackend

        points = 40
        laplacian = sp.random(points, points, density=0.1, random_state=1, format="csr")
        gradient = tuple(sp.random(points, points, density=0.1, random_state=seed, format="csr") for seed in (2, 3, 4))
        converter = DeviceBlochConverter(NumpyArrayBackend())

        def hamiltonian(k_point):
            k_point = np.asarray(k_point, dtype=float)
            return SimpleNamespace(
                negative_laplacian=laplacian,
                gradient=gradient,
                k_point=k_point,
                _k_gradient=sum(value * matrix for value, matrix in zip(k_point, gradient)).tocsr(),
                _diagonal=np.zeros(points),
                nonlocal_operator=SimpleNamespace(projectors=sp.csr_matrix((points, 0)), signs=np.zeros(0)),
            )

        block = np.random.default_rng(0).standard_normal((points, 2)) + 0j
        first = converter.operator(hamiltonian([0.1, 0.2, 0.0])).apply(block)
        for _ in range(5):  # five more SCF steps of the same k-point
            again = converter.operator(hamiltonian([0.1, 0.2, 0.0])).apply(block)
        np.testing.assert_allclose(again, first)
        converter.operator(hamiltonian([0.3, 0.0, 0.0]))
        # the Laplacian and one k.grad per k-point
        self.assertEqual(len(converter._static), 3)


@unittest.skipUnless(cupy_available(), "CuPy and a CUDA device are required")
class BlochExchangeSeveralDevicesTests(unittest.TestCase):
    """The tiles of one build over every device of the node (skipped with fewer than two)."""

    def test_devices_agree_with_one_device(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import (
            DevicePeriodicPairSolvers,
            apply_device_bloch_exchange,
            compress_device_bloch_exchange,
            device_bloch_channel,
        )

        count = cp.cuda.runtime.getDeviceCount()
        if count < 2:
            self.skipTest("one CUDA device")
        from parsec_python.acceleration.Hamiltonian import bloch_exchange_device as module

        devices = tuple(range(count))
        case = _Case(mesh=(3, 2, 1), states=6)
        results = []
        small = module.SMALL_BUILD_POINTS
        module.SMALL_BUILD_POINTS = 0.0  # this small build on every device all the same
        self.addCleanup(setattr, module, "SMALL_BUILD_POINTS", small)
        for owners in ((0,), devices):
            orbitals = []
            for index, vectors in enumerate(case.orbitals):
                with cp.cuda.Device(owners[index % len(owners)]):
                    orbitals.append(cp.asarray(vectors))
            channel = device_bloch_channel(case.orbital_set(orbitals), case.volume_element)
            solvers = DevicePeriodicPairSolvers(case.kernel, case.box(), 4)
            application = apply_device_bloch_exchange(channel, solvers, owners)
            applied = [cp.asnumpy(values) for values in application.applied]
            operators = compress_device_bloch_exchange(application, fraction=1.0)
            for rep, operator in enumerate(operators):
                self.assertEqual(int(operator.projectors.device.id), owners[rep % len(owners)])
            results.append((application.energy, applied))
        self.assertAlmostEqual(results[0][0], results[1][0], delta=1e-13 * abs(results[0][0]))
        for one, several in zip(results[0][1], results[1][1]):
            np.testing.assert_allclose(several, one, rtol=0, atol=1e-13 * np.abs(one).max())


if __name__ == "__main__":
    unittest.main()
