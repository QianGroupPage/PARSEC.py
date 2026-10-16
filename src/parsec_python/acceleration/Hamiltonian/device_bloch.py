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
        return (2 if self.spinor else 1) * self.grid_size

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dimension, self.dimension)

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
        real = block.view(xp.float64)
        # cuSPARSE returns a column-major product; the complex view below needs
        # row-major data, so make it so (a no-op for NumPy, which already is).
        result = xp.ascontiguousarray(self.laplacian @ real).view(xp.complex128)
        if self.k_gradient is not None:
            result -= 2j * xp.ascontiguousarray(self.k_gradient @ real).view(xp.complex128)
        result += self.diagonal[:, None] * block
        if self.projector_forward is not None:
            if self.projector_real:
                coefficients = self._real_product(self.projector_adjoint, block)
                coefficients *= self.projector_signs[:, None]
                result += self._real_product(self.projector_forward, coefficients)
            else:
                coefficients = self.projector_adjoint @ block
                coefficients *= self.projector_signs[:, None]
                result += self.projector_forward @ coefficients
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
        q_up = self._scalar(up)
        q_down = self._scalar(down)
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
        return operator


# ------------------------------------------------------------------- conversion
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
        self._static: dict[int, tuple[Any, Any]] = {}  # id(host object) -> (host ref, device data)
        # PARSEC_KPOINT_SOLVER=host keeps the host eigensolver (only the filter on the device)
        self.device_solver = os.environ.get("PARSEC_KPOINT_SOLVER", "device").strip().lower() != "host"

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

    def _scalar_parts(self, hamiltonian):
        backend = self.backend
        laplacian = self._cached(
            hamiltonian.negative_laplacian,
            lambda: backend.matrix(hamiltonian.negative_laplacian, dtype=np.float64),
        )
        k_gradient = None
        if hamiltonian._k_gradient is not None:
            k_gradient = self._cached(
                hamiltonian._k_gradient,
                lambda: backend.matrix(hamiltonian._k_gradient, dtype=np.float64),
            )
        diagonal = backend.asarray(np.asarray(hamiltonian._diagonal, dtype=np.float64))
        nonlocal_operator = hamiltonian.nonlocal_operator
        projector_real = False
        if nonlocal_operator.projectors.shape[1] == 0:
            forward = adjoint = signs = None
        else:
            # Gamma-point and cluster projectors are real: keep them real so that the
            # products run on the real view of the block (half the work of complex ones).
            projector_real = not np.iscomplexobj(nonlocal_operator.projectors) or bool(
                np.all(nonlocal_operator.projectors.data.imag == 0)
            )
            matrix_dtype = np.float64 if projector_real else np.complex128
            forward, adjoint, signs = self._cached(
                nonlocal_operator,
                lambda: (
                    backend.matrix(nonlocal_operator.projectors, dtype=matrix_dtype),
                    backend.matrix(nonlocal_operator.projectors.conj().T, dtype=matrix_dtype),
                    backend.asarray(np.asarray(nonlocal_operator.signs, dtype=np.float64)),
                ),
            )
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
            forward = sp.coo_matrix(
                (values.real if real else values, (np.concatenate(rows), np.concatenate(cols))),
                shape=(grid_size, 16 * len(soc_projectors)),
            ).tocsr()
            return _BatchedSOC(
                forward=backend.matrix(forward, dtype=dtype),
                adjoint=backend.matrix(forward.conj().T, dtype=dtype),
                sign=backend.asarray(np.asarray(signs, dtype=np.float64)[:, :, None]),
                atoms=len(soc_projectors),
                real=real,
            )

        return self._cached(soc_projectors, build)

    def operator(self, hamiltonian) -> DeviceBlochOperator:
        """Device operator for a host ``KPointKohnShamHamiltonian`` or spinor one."""
        backend = self.backend
        backend.use()
        scalar = getattr(hamiltonian, "scalar_hamiltonian", None)
        spinor = scalar is not None
        parts = self._scalar_parts(scalar if spinor else hamiltonian)
        soc, delta, field = None, None, None
        if spinor:
            soc = self._soc_parts(hamiltonian.soc_projectors, int(parts[2].shape[0]))
            if hamiltonian.xc_delta is not None:
                delta = backend.asarray(np.asarray(hamiltonian.xc_delta, dtype=np.float64))
            if hamiltonian.xc_field is not None:
                field = backend.asarray(np.asarray(hamiltonian.xc_field, dtype=np.float64))
        laplacian, k_gradient, diagonal, forward, adjoint, signs, projector_real = parts
        return DeviceBlochOperator(
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
        )


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
