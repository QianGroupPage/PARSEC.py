"""Device-resident Bloch Hamiltonians for k-point and band calculations.

A periodic Kohn-Sham problem at one k-point is applied here on an *array
backend*: NumPy/SciPy (the reference and the test double) or CuPy on one GPU.
The operator is the same one the CPU code applies

``H_k = -nabla^2 - 2i k.grad + |k|^2 + V + V_NL,k``    (scalar), or
``[H_k (x) 1_spin + B_xc.sigma + V_SO]``                (spinor),

built from the host Hamiltonian objects (:class:`KPointKohnShamHamiltonian`,
:class:`KPointSpinorKohnShamHamiltonian`): the Laplacian, the pre-summed
``k.grad`` (both real, applied to the real view of the complex block), the
Bloch-phase nonlocal projectors, the spin-orbit projector pairs and the local
fields.  Static matrices are uploaded once per device and shared between
k-points; only the local potential and fields are uploaded per SCF step.

The Chebyshev filter dominates the cost of every eigensolver here, so the
operator also provides :meth:`DeviceBlochOperator.chebyshev_filter`, the whole
degree-``m`` recurrence on the device (one upload and one download per
block).  :func:`parsec_python.Eigensolvers.chebyshev.chebyshev_filter` calls it
through the ``device_chebyshev_filter`` attribute of the operator, so CHEBFF,
CHEBDAV and the SUBSPACE filter run unchanged, with their small dense algebra
on the host.

Only operations that NumPy and CuPy share are used, so the NumPy backend runs
exactly the code the GPU runs; the tests rely on that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

BACKEND_ENV = "PARSEC_KPOINT_BACKEND"  # explicit override: cpu | cupy | auto | numpy-device
DEVICES_ENV = "PARSEC_CUPY_DEVICES"  # auto | current | 0,1,...


# ---------------------------------------------------------------- array backends
def _cast_matrix(host_matrix: sp.spmatrix, dtype=None) -> sp.csr_matrix:
    """``host_matrix`` as CSR of ``dtype``; a complex matrix with no imaginary part becomes real
    through ``.real`` (a plain ``astype`` would warn about discarding the imaginary part)."""
    matrix = host_matrix.tocsr()
    if dtype is None:
        return matrix
    if np.issubdtype(np.dtype(dtype), np.floating) and np.iscomplexobj(matrix.data):
        return matrix.real.tocsr().astype(dtype)
    return matrix.astype(dtype)


class NumpyArrayBackend:
    """Host arrays through the device interface (reference and test double)."""

    name = "numpy-device"

    def __init__(self) -> None:
        self.xp = np
        self.device_id = 0

    def use(self) -> None:
        return None

    def matrix(self, host_matrix: sp.spmatrix, dtype=None):
        return _cast_matrix(host_matrix, dtype)

    def asarray(self, host_array):
        return np.ascontiguousarray(host_array)

    def asnumpy(self, array) -> np.ndarray:
        return np.asarray(array)

    def synchronize(self) -> None:
        return None


class CupyArrayBackend:
    """One CUDA device through CuPy."""

    name = "cupy"

    def __init__(self, device_id: int) -> None:
        from ..backends.cupy import require_cupy

        self.cp, self.cpsparse = require_cupy()
        self.xp = self.cp
        self.device_id = int(device_id)

    def use(self) -> None:
        # The current device is per thread: every worker thread calls this.
        self.cp.cuda.Device(self.device_id).use()

    def matrix(self, host_matrix: sp.spmatrix, dtype=None):
        return self.cpsparse.csr_matrix(_cast_matrix(host_matrix, dtype))

    def asarray(self, host_array):
        return self.cp.asarray(np.ascontiguousarray(host_array))

    def asnumpy(self, array) -> np.ndarray:
        return self.cp.asnumpy(array)

    def synchronize(self) -> None:
        self.cp.cuda.Device(self.device_id).synchronize()


# ------------------------------------------------------- spin-orbit ladder algebra
def _ladder_and_lz():
    from ...Hamiltonian.spinor_operator import (
        _LADDER_FROM_DOWN,
        _LADDER_FROM_UP,
        _LZ_HALF,
        _QUARTER_LL1,
    )
    from ...Eigensolvers.perturbative_soc import _LM_TO_L

    return _LADDER_FROM_DOWN, _LADDER_FROM_UP, _LZ_HALF, _QUARTER_LL1, _LM_TO_L


@dataclass
class _BatchedSOC:
    """The spin-orbit projector pairs of every atom as two sparse matrices.

    ``forward`` is ``(n, 16*atoms)``: the columns ``16*a : 16*a + 16`` of atom
    ``a`` hold ``[v_ion | v_so]`` on its support rows.  ``adjoint`` is its
    conjugate transpose.  One sparse product then gives the overlaps of every
    atom with a block, the ladder algebra runs on ``(atoms, 8, vectors)``
    arrays, and one more sparse product (which also sums the overlapping
    supports) adds the result back: a handful of kernels per application in
    place of a Python loop over the atoms.
    """

    forward: Any
    adjoint: Any
    sign: Any  # (atoms, 8, 1) real: KB sign per harmonic
    atoms: int
    real: bool  # forward/adjoint are real: apply them to the real view


class DeviceBlochOperator:
    """Matrix-free Bloch Hamiltonian (scalar or spinor) on an array backend.

    Instances are built by :class:`DeviceBlochConverter` from host
    Hamiltonians.  ``apply`` works on device arrays of shape ``(dim, vectors)``
    (complex128, C-contiguous); :meth:`as_linear_operator` exposes the host
    interface the eigensolvers use.
    """

    # The orbital terms of a meta-GGA, ``(forward, backward, mix, vtau, k, flux)``:
    # the device first derivatives L_i along the grid directions, the matrices of
    # L_i^T (``None`` where L_i^T = -L_i, then L_i is applied and negated), the
    # 3x3 ``mix`` with G_c = sum_i mix[c, i] L_i (the identity on an
    # orthorhombic grid, L = G) with its copy on the device, vtau on this
    # device, the k-point and the three fields W_c of the orbital form of the
    # density gradient (``None`` in the density form)
    # (DeviceBlochConverter._kinetic_parts); None without a meta-GGA.
    kinetic = None
    # Gershgorin's bound of those terms (Ry), set with them: the filter's upper
    # bound takes it in (chebff_spinor._lanczos_upper_bound, orbital_term_margin).
    spectral_margin = 0.0
    # The local term as one kernel on an orthorhombic periodic grid (.bloch_stencil.BlochStencil) with the
    # diagonal it adds (c_0 + V + |k|^2), set by a converter with ``fused_stencil``; None: the sparse products.
    stencil = None
    stencil_diagonal = None

    def __init__(
        self,
        backend,
        *,
        grid_size: int,
        spinor: bool,
        laplacian,
        k_gradient,
        diagonal,
        projector_forward,
        projector_adjoint,
        projector_signs,
        soc: "_BatchedSOC | None",
        delta=None,
        field=None,
        projector_real: bool = False,
        halo_begin=None,
        halo_finish=None,
        laplacian_halo=None,
        k_gradient_halo=None,
        allsum=None,
        global_grid_size: int | None = None,
        mpi=None,
        row_bounds: tuple[int, int] | None = None,
        hubbard=None,
        exchange=None,
    ) -> None:
        self.backend = backend
        self.xp = backend.xp
        self.grid_size = grid_size
        self.spinor = spinor
        self.laplacian = laplacian
        self.k_gradient = k_gradient
        self.diagonal = diagonal
        self.projector_forward = projector_forward
        self.projector_adjoint = projector_adjoint
        self.projector_signs = projector_signs
        self.soc = soc
        self.projector_real = projector_real
        self.delta = delta
        self.field = field
        # A basis distributed over MPI ranks by blocks of grid rows: ``grid_size`` counts the rows of
        # this rank.  The stencils reach rows of the neighbours: ``halo_begin(block)`` starts the exchange
        # of those rows, ``halo_finish(pending)`` returns them, and ``laplacian_halo`` / ``k_gradient_halo``
        # are the parts of the stencil matrices on those rows; the part on the local rows is applied while
        # the exchange is under way.  ``allsum(array)`` sums the overlaps with the projectors over the
        # ranks.  Serially none of these exists.
        self.halo_begin = halo_begin
        self.halo_finish = halo_finish
        self.laplacian_halo = laplacian_halo
        self.k_gradient_halo = k_gradient_halo
        self.allsum = allsum
        self.mpi = mpi  # the MPI context of a distributed basis, None serially
        self.row_bounds = row_bounds  # rows [start, stop) of the grid held here
        # LDA+U of the channel (plusu.f90z): ``(forward, adjoint, potentials)`` with the Bloch
        # projectors P_k (grid x 5 atoms), P_k^H and the device ``D`` (atoms, 5, 5); the scalar
        # operator adds P_k D P_k^H.  None without LDA+U (DeviceBlochConverter._hubbard_parts).
        self.hubbard = hubbard
        # The exact exchange of a hybrid functional (adaptively compressed, one k-point of one channel):
        # ``(xi, factor)`` with ``xi`` (grid x rank, complex, F order) on this device and ``factor = -a_x``;
        # the scalar operator adds ``factor xi (xi^H X)`` (DeviceBlochConverter._exchange_parts).
        self.exchange = exchange
        self.global_grid_size = grid_size if global_grid_size is None else int(global_grid_size)
        xp = self.xp
        if field is not None:
            self._bz = field[2][:, None]
            self._b_minus = (field[0] - 1j * field[1])[:, None]
            self._b_plus = (field[0] + 1j * field[1])[:, None]
        if soc is not None:
            ladder_down, ladder_up, lz_half, quarter_ll1, lm_to_l = _ladder_and_lz()
            # complex so every matmul below is complex @ complex
            self._ladder_from_down = xp.asarray(ladder_down).astype(xp.complex128)
            self._ladder_from_up = xp.asarray(ladder_up).astype(xp.complex128)
            self._lz_half = xp.asarray(lz_half)[:, None]
            self._quarter_ll1 = xp.asarray(quarter_ll1)[:, None]

    @property
    def dimension(self) -> int:
        """Rows of the whole operator (of the whole grid, whichever rows this rank holds)."""
        return (2 if self.spinor else 1) * self.global_grid_size

    @property
    def local_dimension(self) -> int:
        """Rows of the blocks this operator is applied to."""
        return (2 if self.spinor else 1) * self.grid_size

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dimension, self.dimension)

    def local_vector(self, global_vector: np.ndarray) -> np.ndarray:
        """The entries of a host vector of the whole operator (spinor: up then down) held here."""
        if self.row_bounds is None:
            return global_vector
        start, stop = self.row_bounds
        n = self.global_grid_size
        if not self.spinor:
            return global_vector[start:stop]
        return np.concatenate((global_vector[start:stop], global_vector[n + start : n + stop]))

    def as_eigensolver_operator(self) -> "DeviceBlochOperator":
        """The device-resident operator itself (``apply`` on device blocks)."""
        return self

    def _real_product(self, matrix, block):
        """``matrix @ block`` for a real sparse ``matrix`` and a complex block.

        The real view ``(rows, 2*vectors)`` halves the work of a complex product;
        cuSPARSE returns a column-major result, which the complex view needs
        row-major (a no-op for NumPy).
        """
        xp = self.xp
        return xp.ascontiguousarray(matrix @ block.view(xp.float64)).view(xp.complex128)

    # ------------------------------------------------------------ the operator
    def _scalar(self, block):
        """``H_k`` on one ``(n, vectors)`` complex block."""
        xp = self.xp
        # the stencils reach rows of the neighbouring ranks when the basis is distributed: start the
        # exchange, apply the stencils to the local rows meanwhile, then add the part on the halo rows
        pending = None if self.halo_begin is None else self.halo_begin(block)
        if self.stencil is not None and pending is None:
            # -nabla^2 - 2i k.grad + V + |k|^2 in one kernel (orthorhombic periodic grid)
            result = self.stencil.apply(block, self.stencil_diagonal)
        else:
            real = block.view(xp.float64)
            # cuSPARSE returns a column-major product; the complex view below needs
            # row-major data, so make it so (a no-op for NumPy, which already is).
            result = xp.ascontiguousarray(self.laplacian @ real).view(xp.complex128)
            if self.k_gradient is not None:
                result -= 2j * xp.ascontiguousarray(self.k_gradient @ real).view(xp.complex128)
            if pending is not None:
                halo_real = self.halo_finish(pending).view(xp.float64)
                if self.laplacian_halo is not None:
                    result += xp.ascontiguousarray(self.laplacian_halo @ halo_real).view(xp.complex128)
                if self.k_gradient_halo is not None:
                    result -= 2j * xp.ascontiguousarray(self.k_gradient_halo @ halo_real).view(xp.complex128)
            result += self.diagonal[:, None] * block
        if self.kinetic is not None:
            result += self._kinetic_term(block)
        if self.projector_forward is not None:
            if self.projector_real:
                coefficients = self._real_product(self.projector_adjoint, block)
            else:
                coefficients = self.projector_adjoint @ block
            if self.allsum is not None:
                coefficients = self.allsum(coefficients)
            coefficients *= self.projector_signs[:, None]
            if self.projector_real:
                result += self._real_product(self.projector_forward, coefficients)
            else:
                result += self.projector_forward @ coefficients
        if self.hubbard is not None:
            # sum_a sum_(m,m') |phi_a,m> D_(m,m') <phi_a,m'|psi>, the term of
            # parsec_python.SCF.kpoint_hubbard.BlochHubbardHamiltonian
            forward, adjoint, potentials = self.hubbard
            overlaps = adjoint @ block
            if self.allsum is not None:
                overlaps = self.allsum(overlaps)
            atoms = int(potentials.shape[0])
            mixed = xp.matmul(potentials, overlaps.reshape(atoms, 5, -1))
            result += forward @ mixed.reshape(5 * atoms, -1)
        if self.exchange is not None:
            # a_x V_ACE = -a_x xi xi^H (parsec_python.Hamiltonian.kpoint_operator.KPointExchangeHamiltonian)
            from .bloch_exchange_device import adjoint_product

            projectors, factor = self.exchange
            coefficients = adjoint_product(xp, projectors, block)
            if xp is np:
                result += factor * (projectors @ coefficients)
            else:
                from cupy import cublas

                cublas.gemm("N", "N", projectors, coefficients, result, factor, 1.0)
        return result

    @property
    def fused_recurrence(self) -> bool:
        """Whether :meth:`chebyshev_step` takes a step of the filter recurrence in the stencil kernel: the
        kernel of the local term and no term besides the projectors (no meta-GGA, LDA+U or exchange)."""
        return (
            self.stencil is not None
            and self.halo_begin is None
            and not self.spinor
            and self.kinetic is None
            and self.hubbard is None
            and self.exchange is None
        )

    def chebyshev_step(self, current, previous, center: float, scale: float, before=None, after=None):
        """``((H current - center current) scale - before previous) after`` (``before``, ``after`` per
        column; ``previous`` may be ``None``): the projector products, then one pass of the stencil kernel
        (:meth:`.bloch_stencil.BlochStencil.apply`).  Only where :attr:`fused_recurrence`."""
        xp = self.xp
        nonlocal_part = None
        if self.projector_forward is not None:
            if self.projector_real:
                coefficients = self._real_product(self.projector_adjoint, current)
            else:
                coefficients = self.projector_adjoint @ current
            coefficients *= self.projector_signs[:, None]
            if self.projector_real:
                nonlocal_part = self._real_product(self.projector_forward, coefficients)
            else:
                nonlocal_part = xp.ascontiguousarray(self.projector_forward @ coefficients)
        return self.stencil.apply(
            current,
            self.stencil_diagonal,
            added=nonlocal_part,
            previous=previous,
            center=center,
            scale=scale,
            before=before,
            after=after,
        )

    def _kinetic_term(self, block):
        """The meta-GGA terms of a complex block: ``sum_c (G_c + i k_c)^H diag(vtau) (G_c + i k_c)`` and,
        in the orbital form, ``2 sum_c [W_c G_c + G_c^T W_c]``.

        :meth:`parsec_python.Hamiltonian.kpoint_operator.KPointKohnShamHamiltonian.apply_kinetic_potential`
        on the device, with the derivatives along the grid directions,
        ``G_c = sum_i mix[c, i] L_i`` (on an oblique grid a third of the
        entries of a Cartesian ``G_c``): three products ``L_i u`` on the real
        view of the block, the pointwise part in one kernel
        (:func:`.bloch_meta_gga_device.pointwise_terms`) and three products
        ``L_i^T b_i``.
        """
        from .bloch_meta_gga_device import pointwise_terms

        forward, backward, (mix, device_mix), vtau, k_point, fluxes = self.kinetic
        if hasattr(forward[0], "product"):
            # CuPy: the strided CSR kernel of the meta-GGA (DeviceCSR) on the real
            # views, no layout conversions, the transposed products accumulated
            # into the result.
            xp = self.xp
            real = block.view(xp.float64)
            along = [matrix.product(real).view(xp.complex128) for matrix in forward]
            outgoing, result = pointwise_terms(
                xp, along, block, vtau, fluxes, mix, k_point, device_mix=device_mix
            )
            target = result.view(xp.float64)
            for axis in range(3):
                backward[axis].product(outgoing[axis].view(xp.float64), target, accumulate=True)
            return result
        along = [self._real_product(matrix, block) for matrix in forward]
        outgoing, result = pointwise_terms(
            self.xp, along, block, vtau, fluxes, mix, k_point, device_mix=device_mix
        )
        for axis in range(3):
            if backward[axis] is None:
                result -= self._real_product(forward[axis], outgoing[axis])
            else:
                result += self._real_product(backward[axis], outgoing[axis])
        return result

    def _spin_orbit(self, block):
        """``lzsz + lsxy`` on the stacked spinor block (see ``apply_spin_orbit``)."""
        xp = self.xp
        n = self.grid_size
        vectors = block.shape[1]
        soc = self.soc
        atoms = soc.atoms
        both = xp.concatenate((block[:n], block[n:]), axis=1)  # (n, 2*vectors): [up | down]
        overlaps = (
            self._real_product(soc.adjoint, both) if soc.real else soc.adjoint @ both
        )  # (16*atoms, 2*vectors)
        if self.allsum is not None:
            overlaps = self.allsum(overlaps)
        overlaps = overlaps.reshape(atoms, 16, 2, vectors)
        io_up, so_up = soc.sign * overlaps[:, :8, 0], soc.sign * overlaps[:, 8:, 0]
        io_dn, so_dn = soc.sign * overlaps[:, :8, 1], soc.sign * overlaps[:, 8:, 1]
        coefficients = xp.empty((atoms, 16, 2, vectors), dtype=xp.complex128)
        for spin, (sigma, io_same, so_same, ladder, io_other, so_other) in enumerate(
            (
                (1.0, io_up, so_up, self._ladder_from_down, io_dn, so_dn),
                (-1.0, io_dn, so_dn, self._ladder_from_up, io_up, so_up),
            )
        ):
            lz = sigma * self._lz_half
            coefficients[:, :8, spin] = lz * so_same + xp.matmul(ladder, so_other)
            coefficients[:, 8:, spin] = (
                lz * io_same
                + (self._quarter_ll1 - 0.5 * lz) * so_same
                + xp.matmul(ladder, io_other - 0.5 * so_other)
            )
        coefficients = coefficients.reshape(16 * atoms, 2 * vectors)
        contribution = (
            self._real_product(soc.forward, coefficients) if soc.real else soc.forward @ coefficients
        )  # (n, 2*vectors)
        return xp.concatenate((contribution[:, :vectors], contribution[:, vectors:]), axis=0)

    def apply(self, block):
        """``H`` applied to a device block ``(dim, vectors)``."""
        if not self.spinor:
            return self._scalar(block)
        n = self.grid_size
        up, down = block[:n], block[n:]
        # the two components see the same scalar operator: apply it to both as one block of twice the
        # columns (one stencil product, one halo exchange, one projector sum instead of two of each)
        both = self.xp.concatenate((up, down), axis=1)
        scalar_part = self._scalar(both)
        vectors = block.shape[1]
        q_up, q_down = scalar_part[:, :vectors], scalar_part[:, vectors:]
        if self.delta is not None:
            q_up += self.delta[:, None] * up
            q_down -= self.delta[:, None] * down
        if self.field is not None:
            q_up += self._bz * up + self._b_minus * down
            q_down += self._b_plus * up - self._bz * down
        result = self.xp.concatenate([q_up, q_down], axis=0)
        if self.soc is not None:
            result += self._spin_orbit(block)
        return result

    # ------------------------------------------------------- host-facing views
    def matmat(self, host_block: np.ndarray) -> np.ndarray:
        """Host in, host out: ``H @ block`` computed on the device."""
        host_block = np.asarray(host_block, dtype=complex)
        single = host_block.ndim == 1
        if single:
            host_block = host_block[:, None]
        result = self.backend.asnumpy(self.apply(self.backend.asarray(host_block)))
        return result[:, 0] if single else result

    def chebyshev_filter(
        self,
        host_block: np.ndarray,
        degree: int,
        lower_bound: float,
        upper_bound: float,
        reference_eigenvalue: float,
        initial_sigma: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """PARSEC's normalized Chebyshev recurrence, entirely on the device.

        The same recurrence, in the same operations, as
        :func:`parsec_python.Eigensolvers.chebyshev.chebyshev_filter`; returns
        the filtered block (host) and the final ``sigma``.
        """
        xp = self.xp
        block = self.backend.asarray(np.asarray(host_block, dtype=complex))
        half_span = 0.5 * (upper_bound - lower_bound)
        center = 0.5 * (upper_bound + lower_bound)
        sigma_one = half_span / (reference_eigenvalue - center)
        sigma = sigma_one if initial_sigma is None else float(initial_sigma)
        previous = block.copy()
        current = (self.apply(block) - center * block) * (sigma_one / half_span)
        for _ in range(2, degree + 1):
            sigma_next = 1.0 / (2.0 / sigma_one - sigma)
            following = (
                (self.apply(current) - center * current) * (2.0 / half_span) - sigma * previous
            )
            following *= sigma_next
            previous, current = current, following
            sigma = sigma_next
        return self.backend.asnumpy(current), float(sigma)

    def as_linear_operator(self) -> LinearOperator:
        operator = LinearOperator(
            (self.dimension, self.dimension),
            matvec=self.matmat,
            matmat=self.matmat,
            rmatvec=self.matmat,
            dtype=complex,
        )
        operator.device_chebyshev_filter = self.chebyshev_filter
        if self.kinetic is not None:
            operator.spectral_margin = self.spectral_margin
        return operator


# ------------------------------------------------------------------- conversion
def _lattice_factors(gradient):
    """``(L, mix)`` with ``G_c = sum_i mix[c, i] L_i``: the first derivatives along the grid directions.

    On an oblique grid every Cartesian ``G_c`` holds the entries of three
    grid directions; the entries ``(r, s)`` of the three matrices together
    form a 3-vector parallel to the Cartesian direction ``mix[:, i]`` of the
    grid direction ``i`` of the pair.  The entries are sorted by that
    direction (at most three), which gives ``L_i`` and ``mix`` without asking
    the grid how it built ``G``; checked to reproduce ``G``.  ``mix`` is
    ``None`` (and ``L = G``) where the matrices are already per direction (an
    orthorhombic grid) or the sorting does not apply.
    """
    matrices = [sp.csr_matrix(matrix, dtype=np.float64) for matrix in gradient]
    union = (abs(matrices[0]) + abs(matrices[1]) + abs(matrices[2])).tocoo()
    if union.nnz == 0:
        return matrices, None
    rows, columns = union.row, union.col
    vectors = np.stack([np.asarray(matrix[rows, columns]).reshape(-1) for matrix in matrices], axis=1)
    if np.all(np.count_nonzero(vectors, axis=1) <= 1):
        return matrices, None  # every entry belongs to one Cartesian derivative: already per direction
    norms = np.linalg.norm(vectors, axis=1)
    units = vectors / norms[:, None]
    # one sign per direction: its first component of magnitude above 1e-8 positive
    leading = np.argmax(np.abs(units) > 1e-8, axis=1)
    signs = np.sign(units[np.arange(units.shape[0]), leading])
    units *= signs[:, None]
    directions, labels = np.unique(np.round(units, 9), axis=0, return_inverse=True)
    labels = np.asarray(labels).reshape(-1)
    if directions.shape[0] > 3:
        return matrices, None
    mix = np.zeros((3, 3))
    factors = []
    for index in range(directions.shape[0]):
        chosen = labels == index
        mix[:, index] = units[chosen][0]
        values = norms[chosen] * signs[chosen]
        factors.append(sp.csr_matrix((values, (rows[chosen], columns[chosen])), shape=matrices[0].shape))
    while len(factors) < 3:
        factors.append(sp.csr_matrix(matrices[0].shape))
    for component, matrix in enumerate(matrices):
        difference = sum(mix[component, axis] * factors[axis] for axis in range(3)) - matrix
        error = abs(difference).max() if difference.nnz else 0.0
        if error > 1e-12 * max(1.0, abs(matrix).max()):
            return matrices, None
    return factors, mix


def _host_kinetic_margin(hamiltonian) -> float:
    """Gershgorin's bound of the meta-GGA terms of a host Hamiltonian (its ``spectral_margin``, or its term's)."""
    margin = getattr(hamiltonian, "spectral_margin", None)
    if margin is None:
        term = getattr(hamiltonian, "kinetic", None)
        margin = 0.0 if term is None else getattr(term, "spectral_margin", 0.0)
    return float(margin or 0.0)


def _host_kinetic_term(hamiltonian):
    """``(gradient matrices, vtau, k)`` of the meta-GGA term of a host Hamiltonian, ``None`` without one.

    A Bloch Hamiltonian carries ``kinetic_potential`` with its ``gradient``
    and ``k_point``; an isolated or Gamma-point
    :class:`parsec_python.Hamiltonian.KohnShamHamiltonian` carries the term
    as ``kinetic`` (:class:`parsec_python.V_xc.meta_gga.KineticPotentialOperator`).
    """
    vtau = getattr(hamiltonian, "kinetic_potential", None)
    if vtau is not None and getattr(hamiltonian, "gradient", None) is not None:
        return (
            tuple(hamiltonian.gradient),
            vtau,
            getattr(hamiltonian, "k_point", None),
            getattr(hamiltonian, "gradient_flux", None),
        )
    term = getattr(hamiltonian, "kinetic", None)
    if term is not None:
        return (
            tuple(term.gradient.matrices),
            term.kinetic_potential,
            term.k_point,
            getattr(term, "gradient_flux", None),
        )
    return None


class DeviceBlochConverter:
    """Turns host Bloch Hamiltonians into device operators, caching static data.

    One converter belongs to one worker (one device).  The Laplacian is shared
    by every k-point; a k-point's projectors and its pre-summed ``k.grad`` are
    uploaded the first time the host objects that carry them are seen (the
    drivers keep those objects alive between SCF steps), so an SCF step only
    uploads the local potential and fields.
    """

    def __init__(self, backend) -> None:
        self.backend = backend
        backend.use()
        # id(host object) -> (host ref, device data); k.grad by (gradient operators, k-point)
        self._static: dict[Any, tuple[Any, Any]] = {}
        # id(exchange term) -> (weak reference to the term, device xi): host projectors uploaded once
        self._exchange_uploads: dict[int, tuple[Any, Any]] = {}
        # PARSEC_KPOINT_SOLVER=host keeps the host eigensolver (only the filter on the device)
        self.device_solver = os.environ.get("PARSEC_KPOINT_SOLVER", "device").strip().lower() != "host"
        # The local term of a scalar Bloch operator on an orthorhombic periodic grid as one CUDA kernel
        # (.bloch_stencil), where the matrices are that stencil; off: the sparse products (the one-node
        # solver keeps them, so that its numbers stay those of before the kernel; the MPI k-point pools set it).
        self.fused_stencil = False
        self._stencils: dict[tuple, Any] = {}

    def eigensolve(self, hamiltonian, number_of_states: int, settings, attempt: int = 0):
        """Lowest ``number_of_states`` pairs of a host Hamiltonian with the device-resident solver.

        The first solve of :func:`~parsec_python.acceleration.Eigensolvers.chebff_spinor.solve_spinor_eigval`
        from a random start: CHEBFF, and for ``initial_method == "chebdav"`` extra CHEBFF cycles to
        CHEBDAV's residual test.  A later ``attempt`` draws another start.  Returns the eigenvalues,
        the host eigenvectors and the largest residual ``||H x - e x||`` of the pairs.  Used by the
        band paths, where every k-point is a first solve.
        """
        import dataclasses

        from ..Eigensolvers.chebff_spinor import solve_spinor_eigval

        if attempt:
            settings = dataclasses.replace(
                settings,
                chebff=dataclasses.replace(
                    settings.chebff, random_seed=settings.chebff.random_seed + attempt
                ),
            )
        operator = self.operator(hamiltonian)
        result = solve_spinor_eigval(
            operator,
            number_of_states,
            settings=settings,
            state=None,
            xp=self.backend.xp,
            sync=self.backend.synchronize,
            converge_first=settings.initial_method == "chebdav",
        )
        residual = float(np.max(result.residual_norms)) if result.residual_norms is not None else float("nan")
        vectors = np.asarray(self.backend.asnumpy(result.vectors))
        return np.asarray(result.eigenvalues, dtype=float), vectors, residual

    def _cached(self, host_object, build):
        entry = self._static.get(id(host_object))
        if entry is None or entry[0] is not host_object:
            entry = (host_object, build())
            self._static[id(host_object)] = entry
        return entry[1]

    def _cached_k_gradient(self, hamiltonian, build):
        """The device ``k.grad`` of a Bloch Hamiltonian, cached by its gradient operators and its k-point.

        The k-point drivers build a new host Hamiltonian, and with it a new
        pre-summed ``k.grad``, for every task in every SCF step; caching by
        that object (as :meth:`_cached` does) kept one host and one device
        copy per task and step, without bound.  The key holds only the
        shared gradient operators and the k-point.
        """
        host = hamiltonian._k_gradient
        gradient = getattr(hamiltonian, "gradient", None)
        k_point = getattr(hamiltonian, "k_point", None)
        if gradient is None or k_point is None:
            return self._cached(host, build)
        anchors = tuple(gradient)
        key = ("k_gradient",) + tuple(id(item) for item in anchors) + tuple(
            float(value) for value in np.asarray(k_point, dtype=np.float64).reshape(-1)
        )
        entry = self._static.get(key)
        if entry is None or any(first is not second for first, second in zip(entry[0], anchors)):
            entry = (anchors, build())
            self._static[key] = entry
        return entry[1]

    # Hooks a row-distributed converter overrides (here: the whole grid on one device).
    def _rows(self, host_matrix):
        """The rows of a ``(grid, columns)`` host matrix this device holds."""
        return host_matrix

    def _local(self, host_array):
        """The entries (last axis) of a per-grid-point host array this device holds."""
        return np.asarray(host_array)

    def _derivative_parts(self, hamiltonian):
        """Device ``(laplacian, k_gradient)``: the stencil matrices, ``k_gradient`` None without Bloch term."""
        backend = self.backend
        laplacian = self._cached(
            hamiltonian.negative_laplacian,
            lambda: backend.matrix(hamiltonian.negative_laplacian, dtype=np.float64),
        )
        # an isolated Hamiltonian has no Bloch term
        host_gradient = getattr(hamiltonian, "_k_gradient", None)
        k_gradient = None
        if host_gradient is not None:
            k_gradient = self._cached_k_gradient(
                hamiltonian,
                lambda: backend.matrix(host_gradient, dtype=np.float64),
            )
        return laplacian, k_gradient

    def _projector_parts(self, nonlocal_operator):
        """Device ``(forward, adjoint, signs, real)`` of the Kleinman-Bylander projectors."""
        backend = self.backend
        if nonlocal_operator.projectors.shape[1] == 0:
            return None, None, None, False
        # Gamma-point and cluster projectors are real: keep them real so that the
        # products run on the real view of the block (half the work of complex ones).
        projector_real = not np.iscomplexobj(nonlocal_operator.projectors) or bool(
            np.all(nonlocal_operator.projectors.data.imag == 0)
        )
        matrix_dtype = np.float64 if projector_real else np.complex128

        def build():
            rows = self._rows(nonlocal_operator.projectors)
            return (
                backend.matrix(rows, dtype=matrix_dtype),
                backend.matrix(rows.conj().T, dtype=matrix_dtype),
                backend.asarray(np.asarray(nonlocal_operator.signs, dtype=np.float64)),
            )

        forward, adjoint, signs = self._cached(nonlocal_operator, build)
        return forward, adjoint, signs, projector_real

    def _bloch_stencil(self, hamiltonian):
        """The kernel of the local term of ``hamiltonian`` where :attr:`fused_stencil` asks for it and its
        matrices are an orthorhombic periodic stencil (:mod:`.bloch_stencil`), else ``None``."""
        if not self.fused_stencil or self.backend.name != "cupy" or self._operator_extras():
            return None
        gradient = getattr(hamiltonian, "gradient", None)
        laplacian = getattr(hamiltonian, "negative_laplacian", None)
        if gradient is None or laplacian is None or _host_kinetic_term(hamiltonian) is not None:
            return None
        from .bloch_stencil import AxialStencils, BlochStencil, grid_shape_of

        anchors = (laplacian, *tuple(gradient))
        key = ("stencils",) + tuple(id(item) for item in anchors)
        entry = self._stencils.get(key)
        if entry is None or any(first is not second for first, second in zip(entry[0], anchors)):
            shape = grid_shape_of(tuple(gradient), laplacian.shape[0])
            stencils = None if shape is None else AxialStencils.from_matrices(shape, laplacian, tuple(gradient))
            entry = (anchors, stencils, {})
            self._stencils[key] = entry
        _, stencils, by_k = entry
        if stencils is None:
            return None
        k_point = tuple(
            float(value) for value in np.asarray(getattr(hamiltonian, "k_point", np.zeros(3)), dtype=float).reshape(3)
        )
        found = by_k.get(k_point)
        if found is None:
            found = by_k[k_point] = BlochStencil(self.backend.cp, stencils, k_point)
        return found

    def _scalar_parts(self, hamiltonian, *, derivatives: bool = True):
        backend = self.backend
        laplacian, k_gradient = self._derivative_parts(hamiltonian) if derivatives else (None, None)
        # an isolated Hamiltonian keeps its potential as ``effective_potential``
        host_diagonal = getattr(hamiltonian, "_diagonal", None)
        if host_diagonal is None:
            host_diagonal = hamiltonian.effective_potential
        diagonal = backend.asarray(np.asarray(self._local(np.asarray(host_diagonal, dtype=np.float64))))
        forward, adjoint, signs, projector_real = self._projector_parts(hamiltonian.nonlocal_operator)
        return laplacian, k_gradient, diagonal, forward, adjoint, signs, projector_real

    def _soc_parts(self, soc_projectors, grid_size: int) -> "_BatchedSOC | None":
        """All atoms' spin-orbit projectors as one pair of sparse matrices (cached)."""
        if not len(soc_projectors):
            return None
        backend = self.backend

        def build():
            rows, cols, data, signs = [], [], [], []
            for index, atom in enumerate(soc_projectors):
                support = np.asarray(atom.support_rows, dtype=np.int64)
                stacked = np.asarray(atom.stacked)
                rows.append(np.repeat(support, 16))
                cols.append(np.tile(np.arange(16) + 16 * index, support.size))
                data.append(stacked.reshape(-1))
                signs.append(
                    np.where(np.asarray([1, 1, 1, 2, 2, 2, 2, 2]) == 1, atom.sign[0], atom.sign[1])
                )
            values = np.concatenate(data)
            real = not np.iscomplexobj(values) or bool(np.all(values.imag == 0))
            dtype = np.float64 if real else np.complex128
            forward = self._rows(
                sp.coo_matrix(
                    (values.real if real else values, (np.concatenate(rows), np.concatenate(cols))),
                    shape=(grid_size, 16 * len(soc_projectors)),
                ).tocsr()
            )
            return _BatchedSOC(
                forward=backend.matrix(forward, dtype=dtype),
                adjoint=backend.matrix(forward.conj().T, dtype=dtype),
                sign=backend.asarray(np.asarray(signs, dtype=np.float64)[:, :, None]),
                atoms=len(soc_projectors),
                real=real,
            )

        return self._cached(soc_projectors, build)

    def _hubbard_parts(self, term):
        """Device ``(P_k, P_k^H, D)`` of an LDA+U term (``HubbardOperator``).

        The projectors are cached with the host object that holds them (one per k-point, kept by
        :class:`parsec_python.SCF.kpoint_hubbard.KPointHubbard`); ``D`` changes every SCF step and
        is uploaded with the local potential.
        """
        backend = self.backend
        projectors = term.projectors

        def build():
            rows = self._rows(projectors.projectors)
            return (
                backend.matrix(rows, dtype=np.complex128),
                backend.matrix(rows.conj().T, dtype=np.complex128),
            )

        forward, adjoint = self._cached(projectors, build)
        return forward, adjoint, backend.asarray(np.asarray(term.potentials, dtype=np.complex128))

    def _exchange_parts(self, term):
        """Device ``(xi, -a_x)`` of an exchange term (``projectors`` xi, ``fraction`` a_x).

        A device ``xi`` on this device (the device exchange of
        :mod:`.bloch_exchange_device`, built where the k-point's orbitals are)
        is used in place; another device's is copied here and a host ``xi``
        uploaded, once per term (the uploads of terms that no longer exist
        are dropped).
        """
        import weakref

        backend = self.backend
        projectors = term.projectors
        if projectors is None or getattr(term, "released", False):
            raise RuntimeError("the exchange operator of this Hamiltonian was released")
        factor = -float(term.fraction)
        xp = backend.xp
        if xp is not np and isinstance(projectors, xp.ndarray):
            if int(projectors.device.id) == int(backend.device_id) and projectors.flags.f_contiguous:
                return projectors, factor
        elif xp is np and isinstance(projectors, np.ndarray):
            return np.asfortranarray(projectors, dtype=np.complex128), factor
        for key in [key for key, (reference, _) in self._exchange_uploads.items() if reference() is None]:
            del self._exchange_uploads[key]
        found = self._exchange_uploads.get(id(term))
        if found is not None and found[0]() is term:
            return found[1], factor
        if xp is not np and isinstance(projectors, xp.ndarray):
            with xp.cuda.Device(int(projectors.device.id)):
                source = xp.asfortranarray(projectors, dtype=xp.complex128)
            device = xp.empty(source.shape, dtype=xp.complex128, order="F")
            device.data.copy_from_device(source.data, source.nbytes)
        else:
            device = xp.asfortranarray(xp.asarray(np.asarray(projectors, dtype=np.complex128)))
        try:
            self._exchange_uploads[id(term)] = (weakref.ref(term), device)
        except TypeError:  # a term without weak references is uploaded for every operator
            pass
        return device, factor

    def operator(self, hamiltonian) -> DeviceBlochOperator:
        """Device operator for a host ``KPointKohnShamHamiltonian`` or spinor one.

        A k-point spinor Hamiltonian carries its scalar part as ``scalar_hamiltonian``; the spinor
        Hamiltonian of an isolated system (``SpinorKohnShamHamiltonian``) is its own scalar part.
        A scalar Hamiltonian with LDA+U (``BlochHubbardHamiltonian``: ``base`` and ``hubbard``) is
        the operator of its base with the term; one with the exchange of a hybrid functional
        (``KPointExchangeHamiltonian``: ``base`` and ``exchange``) the operator of its base with
        ``-a_x xi xi^H``.
        """
        exchange = getattr(hamiltonian, "exchange", None)
        base = getattr(hamiltonian, "base", None)
        if exchange is not None and base is not None:
            operator = self.operator(base)
            if operator.spinor:
                raise NotImplementedError("the exact exchange of a spinor Hamiltonian is not implemented")
            if operator.row_bounds is not None:
                raise NotImplementedError(
                    "the exact exchange of a hybrid functional needs the orbitals on every grid point "
                    "(not the row-distributed MPI k-point solver)"
                )
            operator.exchange = self._exchange_parts(exchange)
            return operator
        term = getattr(hamiltonian, "hubbard", None)
        if term is not None and base is not None:
            operator = self.operator(base)
            if operator.spinor:
                raise NotImplementedError("LDA+U of a spinor Hamiltonian is not implemented")
            operator.hubbard = self._hubbard_parts(term)
            return operator
        backend = self.backend
        backend.use()
        scalar = getattr(hamiltonian, "scalar_hamiltonian", None)
        spinor = scalar is not None or hasattr(hamiltonian, "soc_projectors")
        host_scalar = scalar if scalar is not None else hamiltonian
        stencil = None if spinor else self._bloch_stencil(host_scalar)
        if stencil is not None:
            # the kernel replaces the stencil matrices: they are not uploaded
            parts = self._scalar_parts(host_scalar, derivatives=False)
        else:
            parts = self._scalar_parts(host_scalar)
        laplacian, k_gradient, diagonal, forward, adjoint, signs, projector_real = parts
        global_size = int(host_scalar.negative_laplacian.shape[0])
        soc, delta, field = None, None, None
        if spinor:
            soc = self._soc_parts(hamiltonian.soc_projectors, global_size)
            if hamiltonian.xc_delta is not None:
                delta = backend.asarray(np.asarray(self._local(np.asarray(hamiltonian.xc_delta, dtype=np.float64))))
            if hamiltonian.xc_field is not None:
                field = backend.asarray(np.asarray(self._local(np.asarray(hamiltonian.xc_field, dtype=np.float64))))
        operator = DeviceBlochOperator(
            backend,
            grid_size=int(diagonal.shape[0]),
            spinor=spinor,
            laplacian=laplacian,
            k_gradient=k_gradient,
            diagonal=diagonal,
            projector_forward=forward,
            projector_adjoint=adjoint,
            projector_signs=signs,
            soc=soc,
            delta=delta,
            field=field,
            projector_real=projector_real,
            global_grid_size=global_size,
            **self._operator_extras(),
        )
        if stencil is not None:
            operator.stencil = stencil
            operator.stencil_diagonal = stencil.diagonal(diagonal)
        term = _host_kinetic_term(host_scalar)
        if term is not None:
            if spinor or operator.row_bounds is not None:
                raise NotImplementedError(
                    "the meta-GGA orbital term is implemented for scalar Bloch Hamiltonians on whole grids "
                    "(not spinors, not the row-distributed MPI k-point solver)"
                )
            operator.kinetic = self._kinetic_parts(*term)
            operator.spectral_margin = _host_kinetic_margin(host_scalar)
        return operator

    def _kinetic_parts(self, gradient, kinetic_potential, k_point, gradient_flux=None):
        """Device ``(L_i, L_i^T or None, mix, vtau, k, W or None)`` of the meta-GGA terms of a Hamiltonian.

        The first-derivative matrices ``gradient`` are shared by the host
        Hamiltonians of a run; their factors along the grid directions
        (:func:`_lattice_factors`) are uploaded once per converter; ``L_i^T``
        is kept only where it is not ``-L_i`` (the centered stencil is
        antisymmetric on periodic, confined and oblique grids alike).
        ``vtau`` and the fluxes ``W`` of the orbital form are uploaded per
        operator.
        """
        backend = self.backend
        gradient = tuple(gradient)

        def upload_all():
            factors, mix = _lattice_factors(gradient)
            uploaded = []
            for matrix in factors:
                host = sp.csr_matrix(matrix, dtype=np.float64)
                if backend.name == "cupy":
                    # the strided CSR kernel of acceleration/Hamiltonian/meta_gga_device.py,
                    # with the transposes stored (deterministic sums)
                    from .meta_gga_device import DeviceCSR

                    uploaded.append((DeviceCSR(backend.cp, host), DeviceCSR(backend.cp, host.T.tocsr())))
                    continue
                skew = (host + host.T).tocsr()
                skew.eliminate_zeros()
                antisymmetric = skew.nnz == 0 or float(np.max(np.abs(skew.data))) == 0.0
                uploaded.append(
                    (
                        backend.matrix(host, dtype=np.float64),
                        None if antisymmetric else backend.matrix(host.T.tocsr(), dtype=np.float64),
                    )
                )
            matrix = np.eye(3) if mix is None else np.asarray(mix, dtype=np.float64)
            return gradient, uploaded, (matrix, backend.asarray(np.ascontiguousarray(matrix.reshape(-1))))

        anchors, pairs, mix = self._cached(gradient[0], upload_all)
        if any(first is not second for first, second in zip(anchors, gradient)):
            anchors, pairs, mix = upload_all()
        vtau = backend.asarray(np.ascontiguousarray(self._local(kinetic_potential), dtype=np.float64))
        k_point = (0.0, 0.0, 0.0) if k_point is None else tuple(
            float(value) for value in np.asarray(k_point, dtype=np.float64).reshape(3)
        )
        fluxes = None
        if gradient_flux is not None:
            fluxes = [
                backend.asarray(np.ascontiguousarray(self._local(flux), dtype=np.float64)) for flux in gradient_flux
            ]
        return (
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            mix,
            vtau,
            k_point,
            fluxes,
        )

    def _operator_extras(self) -> dict:
        """Keyword arguments for the operator that only a distributed converter has."""
        return {}


# --------------------------------------------------------------- backend choice
@dataclass(frozen=True)
class KPointBackend:
    """Resolved execution backend of a k-point/band calculation."""

    name: str  # "cpu", "cupy" or "numpy-device"
    device_ids: tuple[int, ...] = ()

    @property
    def uses_device(self) -> bool:
        return self.name != "cpu"

    def converter(self, worker_index: int) -> DeviceBlochConverter | None:
        """The converter of one worker: device ``device_ids[index % len]``."""
        if self.name == "cpu":
            return None
        if self.name == "numpy-device":
            return DeviceBlochConverter(NumpyArrayBackend())
        device_id = self.device_ids[worker_index % len(self.device_ids)]
        return DeviceBlochConverter(CupyArrayBackend(device_id))


def _cupy_device_ids() -> tuple[int, ...]:
    from ..backends.cupy import cupy_device_count, require_cupy

    cp, _ = require_cupy()
    available = cupy_device_count()
    setting = os.environ.get(DEVICES_ENV, "auto").strip().lower()
    if setting in {"", "auto"}:
        return tuple(range(available))
    if setting in {"current", "off"}:
        return (int(cp.cuda.Device().id),)
    try:
        ids = tuple(dict.fromkeys(int(v.strip()) for v in setting.split(",") if v.strip()))
    except ValueError as error:
        raise ValueError(
            f"{DEVICES_ENV} must be 'auto', 'current' or comma-separated device indices"
        ) from error
    if not ids or any(i < 0 or i >= available for i in ids):
        raise ValueError(f"{DEVICES_ENV} contains an unavailable device")
    return ids


def resolve_kpoint_backend(requested: str | None = None) -> KPointBackend:
    """Backend from ``requested`` or ``PARSEC_KPOINT_BACKEND`` (default ``cpu``).

    ``cupy`` needs CuPy and at least one CUDA device and raises otherwise;
    ``auto`` uses CuPy when it works and the CPU otherwise;  ``numpy-device``
    runs the device code on the host (a test double).
    """
    name = (requested or os.environ.get(BACKEND_ENV, "cpu")).strip().lower()
    if name == "cpu":
        return KPointBackend("cpu")
    if name == "numpy-device":
        return KPointBackend("numpy-device", (0,))
    if name in ("cupy", "auto"):
        try:
            return KPointBackend("cupy", _cupy_device_ids())
        except Exception:
            if name == "cupy":
                raise
            return KPointBackend("cpu")
    raise ValueError(f"{BACKEND_ENV} must be cpu, cupy, auto or numpy-device, got {name!r}")


__all__ = [
    "BACKEND_ENV",
    "CupyArrayBackend",
    "DEVICES_ENV",
    "DeviceBlochConverter",
    "DeviceBlochOperator",
    "KPointBackend",
    "NumpyArrayBackend",
    "resolve_kpoint_backend",
]
