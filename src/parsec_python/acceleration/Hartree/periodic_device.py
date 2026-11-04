"""Hartree potentials of periodic cells, wires and slabs on a CUDA device (CuPy, float64).

The performance counterpart of the readable solves
:func:`parsec_python.Hartree.confined.solve_confined_hartree` (wire and slab:
the boundary values of ``hartset_wire``/``hartset_slab`` and the
conjugate gradients of ``hpotcg``) and
:func:`parsec_python.Hartree.pbc.solve_periodic_hartree` (a 3-D cell: the
reciprocal-space solve of ``Calc_Hartree_G``, and on request the
finite-difference conjugate gradients).  The same equations are solved with
the same algorithm:

* wire: the cylindrical moments ``Q_mn`` of :class:`parsec_python.Hartree.wire.WireBoundaryPlan`
  and the boundary series, as products of the same tables (uploaded once);
* slab: the plane moments ``qc``, ``qs``, ``sigma'`` and the boundary sums of
  :class:`parsec_python.Hartree.slab_fortran.SlabBoundaryPlan`, as matrix products of
  the same tables;
* ``8 pi n - A_IB V_B`` with ``A_IB`` as a sparse matrix of the exterior
  stencil entries, and the unpreconditioned CG recurrence of
  :func:`parsec_python.Hartree.poisson._conjugate_gradient`
  (:class:`FusedConjugateGradient`: same starting vector, tolerance
  ``rtol ||r_0|| + atol``, iteration budget, breakdown rule and final true
  residual, with ``||r|| = sqrt(r.r)`` as NumPy computes it);
* ``Calc_Hartree_G``: ``V_H(G) = 8 pi n(G)/|G|^2`` with the ``|G|^2`` of
  :func:`parsec_python.Hartree.gspace.g_squared` and CuPy's FFT.

Only the order of the floating-point sums differs (BLAS and FFT on the
device), so the results equal the host ones to round-off, not bit for bit.

Which solves run here (``PARSEC_PERIODIC_HARTREE``):

* ``auto`` (default) and ``device``: the wire and slab solves,
  ``Calc_Hartree_G`` and the finite-difference CG of an orthorhombic 3-D
  cell, when the reference drivers put their eigenproblems on a CUDA device
  (:func:`parsec_python.acceleration.SCF.kpoint_parallel.resolve_backend`, i.e.
  ``PARSEC_DEVICE``/``PARSEC_KPOINT_BACKEND``), and so in the accelerated
  driver's Gamma-point path of a cell, which solves with the readable
  periodic CG.  Up to 2026-10-30 the CG of a 3-D cell stayed on the host
  unless ``device`` was named; on the device it stops within the same
  tolerance on another iterate (fixed-order sums), which changes periodic
  results by the CG tolerance (Si8: 1e-9 Ry and less);
* ``host``: every solve on the host (the readable code, exactly; for a plain
  3-D cell the default up to 2026-10-30).

The FFT solve of Fortran's general (non-orthogonal) grid always stays on the
host.  The solve runs on the first device of the k-point plan.  Importing
:mod:`parsec_python.acceleration` installs the two solvers
(:func:`install`).
"""

from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np
import scipy.sparse as sp

from parsec_python.Hartree.confined import (
    ConfinedHartreePlan,
    ConfinedHartreeResult,
    set_confined_hartree_solver,
)
from parsec_python.Hartree.pbc import PeriodicHartreeResult, set_periodic_hartree_solver
from parsec_python.Hartree.slab_fortran import SlabBoundaryPlan
from parsec_python.Hartree.wire import LN2_MINUS_EULER, WireBoundaryPlan

MODE_ENV = "PARSEC_PERIODIC_HARTREE"  # auto (default) | device | host
_MODES = {
    "": "auto",
    "auto": "auto",
    "device": "device",
    "on": "device",
    "1": "device",
    "true": "device",
    "yes": "device",
    "host": "host",
    "off": "host",
    "0": "host",
    "false": "host",
    "no": "host",
}
_CACHE_CAPACITY = 4


def hartree_device_mode() -> str:
    """``auto``, ``device`` or ``host`` from ``PARSEC_PERIODIC_HARTREE``.

    ``auto`` and ``device`` put the same solves on the device (since
    2026-10-30); only the text of :func:`describe_periodic_hartree` tells them
    apart.
    """

    value = os.environ.get(MODE_ENV, "auto").strip().lower()
    if value not in _MODES:
        raise ValueError(f"{MODE_ENV} must be auto, device or host, not {value!r}")
    return _MODES[value]


def hartree_device_id() -> int | None:
    """The CUDA device of the solves, or ``None`` when the reference drivers run on the CPU.

    The first device of the k-point plan (``PARSEC_DEVICE``,
    ``PARSEC_KPOINT_BACKEND``, ``PARSEC_CUPY_DEVICES``); ``numpy-device`` (the
    host test double) and the CPU give ``None``, and so does an MPI run
    (``--mpi``): its ranks run the drivers replicated, which these solves
    leave on the host as before.
    """

    from ..Parallel import mpi
    from ..SCF.kpoint_parallel import resolve_backend

    if mpi.requested():
        return None
    try:
        plan = resolve_backend()
    except Exception:  # noqa: BLE001 - no usable device: the host solves
        return None
    if plan is None or plan.name != "cupy" or not plan.device_ids:
        return None
    return int(plan.device_ids[0])


def _cupy():
    from ..backends.cupy import require_cupy

    return require_cupy()


def _device_scalar(value) -> float:
    item = getattr(value, "item", None)
    return float(item() if item is not None else value)


_KERNEL_FILE = Path(__file__).with_name("periodic_device_cg.cu")
_BLOCK = 256
#: Iterations per batch without a host synchronization.
BATCH_ENV = "PARSEC_PERIODIC_HARTREE_BATCH"
_DEFAULT_BATCH = 8
_KERNELS: dict[int, dict] = {}
_KERNEL_LOCK = threading.Lock()


def _cg_kernels(cp, device_id: int) -> dict:
    """The kernels of ``periodic_device_cg.cu`` on ``device_id`` (one NVRTC compile, cached by CuPy)."""

    with _KERNEL_LOCK:
        kernels = _KERNELS.get(device_id)
        if kernels is None:
            module = cp.RawModule(
                code=_KERNEL_FILE.read_text(encoding="utf-8"), options=("--fmad=false",)
            )
            kernels = {
                name: module.get_function(name)
                for name in (
                    "cg_spmv_dot",
                    "cg_reduce",
                    "cg_update_residual",
                    "cg_update_direction",
                    "cg_apply_solution",
                )
            }
            _KERNELS[device_id] = kernels
        return kernels


def _batch_size() -> int:
    value = os.environ.get(BATCH_ENV, "").strip()
    if not value:
        return _DEFAULT_BATCH
    batch = int(value)
    if batch < 1:
        raise ValueError(f"{BATCH_ENV} must be a positive integer")
    return batch


class FusedConjugateGradient:
    """The CG of :func:`parsec_python.Hartree.poisson._conjugate_gradient` on a device.

    The same recurrence (``alpha = rr/p.Ap``, ``x += alpha p``, ``r -= alpha Ap``,
    ``beta = rr'/rr``, ``p = r + beta p``), starting vector, tolerance
    ``rtol ||r_0|| + atol`` with ``||r|| = sqrt(r.r)`` (what ``numpy.linalg.norm``
    computes for a real vector), matrix-vector budget, breakdown rule (``p.Ap``
    not positive or not finite: stop before the update), early exit and final
    true residual, in the kernels of ``periodic_device_cg.cu``: the CSR product
    summed in row order as SciPy does, dot products as block sums in a fixed
    order (deterministic), and batches of ``PARSEC_PERIODIC_HARTREE_BATCH`` (8)
    iterations without a host synchronization.  The host then reads the
    batch's ``p.Ap`` and ``r.r``, applies the stopping test and the breakdown
    rule iteration by iteration as the reference does, and ``x`` receives the
    accepted iterations' updates in their order (the batch keeps its
    directions for that).  Iterations computed beyond the stop are discarded,
    so the reported iterations, matrix-vector products and solution are those
    of the sequential recurrence.
    """

    def __init__(self, cp, operator, device_id: int, size: int) -> None:
        self.cp = cp
        self.device_id = int(device_id)
        self.size = int(size)
        self.blocks = max(1, (self.size + _BLOCK - 1) // _BLOCK)
        self.kernels = _cg_kernels(cp, self.device_id)
        host = sp.csr_matrix(operator)
        if host.nnz >= 2**31 or host.shape != (self.size, self.size):
            raise ValueError("the operator does not fit the 32-bit CSR kernels of this solver")
        self.indptr = cp.asarray(np.ascontiguousarray(host.indptr, dtype=np.int32))
        self.indices = cp.asarray(np.ascontiguousarray(host.indices, dtype=np.int32))
        self.data = cp.asarray(np.ascontiguousarray(host.data, dtype=np.float64))
        self.partial = cp.empty(self.blocks, dtype=cp.float64)
        self.product = cp.empty(self.size, dtype=cp.float64)
        self.batch = 0
        self.stored = None
        self.den = None
        self.rr = None

    @property
    def device_bytes(self) -> int:
        arrays = [self.indptr, self.indices, self.data, self.partial, self.product]
        arrays += [item for item in (self.stored, self.den, self.rr) if item is not None]
        return int(sum(array.nbytes for array in arrays))

    def _ensure_batch(self, batch: int) -> None:
        if batch != self.batch:
            cp = self.cp
            self.stored = cp.empty((batch, self.size), dtype=cp.float64)
            self.den = cp.empty(batch, dtype=cp.float64)
            self.rr = cp.empty(batch + 1, dtype=cp.float64)
            self.batch = batch

    def multiply(self, vector):
        """``A vector`` (in a buffer of this solver) and the block sums of ``vector . A vector``."""

        self.kernels["cg_spmv_dot"](
            (self.blocks,),
            (_BLOCK,),
            (
                self.indptr,
                self.indices,
                self.data,
                vector,
                self.product,
                self.partial,
                np.int64(self.size),
            ),
        )
        return self.product

    def solve(self, rhs, initial, settings):
        """``(solution, converged, iterations, matvecs, residual_norm, initial_norm)``.

        ``initial`` (a device array of its own) becomes the solution.
        """

        cp = self.cp
        n = np.int64(self.size)
        grid = (self.blocks,)
        block = (_BLOCK,)
        blocks = np.int64(self.blocks)
        solution = initial
        residual = rhs - self.multiply(solution)
        matrix_vector_products = 1
        residual_squared = _device_scalar(cp.vdot(residual, residual))
        initial_norm = float(np.sqrt(residual_squared))
        tolerance = settings.relative_tolerance * initial_norm + settings.absolute_tolerance
        if initial_norm <= tolerance:
            return solution, True, 0, matrix_vector_products, initial_norm, initial_norm
        direction = residual.copy()
        self._ensure_batch(_batch_size())
        self.rr[0] = residual_squared
        iterations = 0
        converged = False
        kernels = self.kernels
        while matrix_vector_products < settings.max_iterations:
            count = min(self.batch, int(settings.max_iterations) - matrix_vector_products)
            for i in range(count):
                operator_direction = self.multiply(direction)
                kernels["cg_reduce"]((1,), block, (self.partial, blocks, self.den, np.int32(i)))
                kernels["cg_update_residual"](
                    grid,
                    block,
                    (residual, operator_direction, self.den, self.rr, np.int32(i), self.partial, n),
                )
                kernels["cg_reduce"]((1,), block, (self.partial, blocks, self.rr, np.int32(i + 1)))
                kernels["cg_update_direction"](
                    grid, block, (direction, residual, self.stored[i], self.rr, np.int32(i), n)
                )
            denominators = cp.asnumpy(self.den[:count])
            squares = cp.asnumpy(self.rr[: count + 1])
            accepted = 0
            stop = False
            for i in range(count):
                matrix_vector_products += 1
                denominator = float(denominators[i])
                if denominator <= 0.0 or not np.isfinite(denominator):
                    stop = True
                    break
                accepted += 1
                iterations += 1
                if float(np.sqrt(squares[i + 1])) <= tolerance:
                    converged = True
                    stop = True
                    break
            if accepted:
                kernels["cg_apply_solution"](
                    grid, block, (solution, self.stored, self.den, self.rr, np.int32(accepted), n)
                )
            if stop:
                break
            # the next batch starts from the last residual of this one
            self.rr[0] = self.rr[count]
        true_residual = rhs - self.multiply(solution)
        residual_norm = float(np.sqrt(_device_scalar(cp.vdot(true_residual, true_residual))))
        matrix_vector_products += 1
        return solution, converged, iterations, matrix_vector_products, residual_norm, initial_norm


def _upload_csr(matrix, cpsparse):
    host = sp.csr_matrix(matrix)
    return cpsparse.csr_matrix(host.astype(np.float64))


class DeviceWireBoundary:
    """The tables of a :class:`~parsec_python.Hartree.wire.WireBoundaryPlan` on the device."""

    def __init__(self, plan: WireBoundaryPlan, cp) -> None:
        self.cp = cp
        order = int(plan.order)
        self.factor = 4.0 * float(plan.volume_element) / float(plan.period)
        self.source_mask = cp.asarray(plan.source_mask, dtype=cp.float64)
        self.source_radial = cp.asarray(plan.source_radial)
        self.source_angular = cp.asarray(plan.source_angular)
        self.source_angular_conjugate = cp.conj(self.source_angular)
        self.source_axial = cp.asarray(plan.source_axial)
        bessel = np.asarray(plan.source_bessel)
        self.source_bessel = cp.asarray(np.ascontiguousarray(bessel))
        self.source_bessel_orders = cp.asarray(np.ascontiguousarray(bessel[:, :, 1:]))
        radius = np.asarray(plan.boundary_radius, dtype=np.float64)
        powers = np.arange(1, order + 1, dtype=np.float64)
        # e^(i m theta) / (2 m rho^m) of every exterior point, as the host
        # expression builds it.
        self.boundary_radial = cp.asarray(
            np.asarray(plan.boundary_angular)
            / (2.0 * powers[None, :] * radius[:, None] ** powers[None, :])
        )
        self.boundary_angular = cp.asarray(plan.boundary_angular)
        self.boundary_angular_conjugate = cp.conj(self.boundary_angular)
        self.boundary_axial = cp.asarray(plan.boundary_axial)
        boundary_bessel = np.asarray(plan.boundary_bessel)
        self.boundary_bessel_zero = cp.asarray(np.ascontiguousarray(boundary_bessel[:, :, 0]))
        self.boundary_bessel_orders = cp.asarray(np.ascontiguousarray(boundary_bessel[:, :, 1:]))
        self.logarithm = cp.asarray(
            (LN2_MINUS_EULER - np.log(radius / float(plan.period))) * 0.5
        )
        self.size = int(np.asarray(plan.source_mask).size)

    @property
    def device_bytes(self) -> int:
        return int(
            sum(
                array.nbytes
                for array in (
                    self.source_mask, self.source_radial, self.source_angular,
                    self.source_angular_conjugate, self.source_axial, self.source_bessel,
                    self.source_bessel_orders, self.boundary_radial, self.boundary_angular,
                    self.boundary_angular_conjugate, self.boundary_axial,
                    self.boundary_bessel_zero, self.boundary_bessel_orders, self.logarithm,
                )
            )
        )

    def values(self, density):
        """``WireBoundaryPlan.boundary_values`` of a device density."""

        cp = self.cp
        factor = self.factor
        weights = density * self.source_mask
        radial = factor * (weights @ self.source_radial)
        phased = weights[:, None] * self.source_axial
        axial = factor * cp.einsum("jn,jn->n", phased, self.source_bessel[:, :, 0])
        # sum_j phased[j, n] I_m(n k rho_j) e^(-+i m theta_j)
        weighted_bessel = phased[:, :, None] * self.source_bessel_orders
        positive = factor * cp.einsum("jnm,jm->nm", weighted_bessel, self.source_angular)
        negative = factor * cp.einsum("jnm,jm->nm", weighted_bessel, self.source_angular_conjugate)
        monopole = factor * cp.sum(density)
        value = (self.boundary_radial * radial[None, :]).sum(axis=1)
        inner = axial[None, :] * self.boundary_bessel_zero
        inner = inner + cp.einsum(
            "bnm,bm->bn", self.boundary_bessel_orders * positive[None, :, :], self.boundary_angular
        )
        inner = inner + cp.einsum(
            "bnm,bm->bn",
            self.boundary_bessel_orders * negative[None, :, :],
            self.boundary_angular_conjugate,
        )
        value = value + cp.sum(self.boundary_axial * inner, axis=1)
        value = value + monopole * self.logarithm
        return 2.0 * value.real


class DeviceSlabBoundary:
    """The tables of a :class:`~parsec_python.Hartree.slab_fortran.SlabBoundaryPlan` on the device."""

    def __init__(self, plan: SlabBoundaryPlan, cp) -> None:
        self.cp = cp
        grid = getattr(plan.to_box, "__self__", None)
        if grid is None or not all(
            hasattr(grid, name) for name in ("integer_coordinates", "index_min", "box_shape")
        ):
            raise TypeError("the slab plan does not name the grid of its index box")
        n1, n2, n3 = (int(value) for value in grid.box_shape)
        self.box_shape = (n1, n2, n3)
        local = np.asarray(grid.integer_coordinates) - np.asarray(grid.index_min)
        flat = (local[:, 0] * n2 + local[:, 1]) * n3 + local[:, 2]
        self.box_index = cp.asarray(flat.astype(np.int64))
        self.size = int(flat.size)
        lengths = np.asarray(plan.lengths, dtype=np.float64)
        self.prefactor = cp.asarray(2.0 * np.pi / (float(plan.area) * lengths))
        count = lengths.size
        self.plane_cos = cp.asarray(np.ascontiguousarray(np.asarray(plan.plane_cos).reshape(count, n1 * n2)))
        self.plane_sin = cp.asarray(np.ascontiguousarray(np.asarray(plan.plane_sin).reshape(count, n1 * n2)))
        counts = np.asarray(plan.plane_counts, dtype=np.float64)
        filled = counts > 0
        # sigma'_k = -2 pi h3 sum_k / count_k for a filled plane, 0 otherwise.
        scale = np.zeros_like(counts)
        scale[filled] = -2.0 * np.pi * float(plan.plane_spacing) / counts[filled]
        self.plane_filled = cp.asarray(filled)
        self.sigma_scale = cp.asarray(scale)
        self.plane_used = cp.asarray(np.asarray(plan.plane_used, dtype=np.float64))
        self.decay = cp.asarray(np.ascontiguousarray(plan.decay))
        self.boundary_plane = cp.asarray(np.asarray(plan.boundary_plane, dtype=np.int64))
        self.boundary_cos = cp.asarray(np.ascontiguousarray(plan.boundary_cos))
        self.boundary_sin = cp.asarray(np.ascontiguousarray(plan.boundary_sin))
        self.distance = cp.asarray(np.ascontiguousarray(plan.distance))
        self.volume_element = float(plan.volume_element)

    @property
    def device_bytes(self) -> int:
        return int(
            sum(
                array.nbytes
                for array in (
                    self.box_index, self.prefactor, self.plane_cos, self.plane_sin,
                    self.plane_filled, self.sigma_scale, self.plane_used, self.decay,
                    self.boundary_plane, self.boundary_cos, self.boundary_sin, self.distance,
                )
            )
        )

    def values(self, density):
        """``SlabBoundaryPlan.boundary_values`` of a device density."""

        cp = self.cp
        n1, n2, n3 = self.box_shape
        box = cp.zeros(n1 * n2 * n3, dtype=cp.float64)
        box[self.box_index] = density
        box = box.reshape(n1 * n2, n3)
        used = self.plane_used
        qc = (self.plane_cos @ box) * self.prefactor[:, None] * used[None, :]
        qs = (self.plane_sin @ box) * self.prefactor[:, None] * used[None, :]
        sigma = cp.where(self.plane_filled, self.sigma_scale * box.sum(axis=0), 0.0) * used
        cosine_part = cp.einsum("gpk,gk->gp", self.decay, qc)
        sine_part = cp.einsum("gpk,gk->gp", self.decay, qs)
        plane = self.boundary_plane
        periodic = cp.sum(self.boundary_cos * cosine_part[:, plane].T, axis=1)
        periodic = periodic + cp.sum(self.boundary_sin * sine_part[:, plane].T, axis=1)
        mean = self.distance @ sigma
        return 2.0 * (mean[plane] + 2.0 * self.volume_element * periodic)


class DeviceConfinedHartree:
    """One wire or slab Hartree problem (operator, exterior stencil, boundary tables) on a device."""

    def __init__(self, plan: ConfinedHartreePlan, negative_laplacian, device_id: int) -> None:
        cp, cpsparse = _cupy()
        self.cp = cp
        self.device_id = int(device_id)
        size = int(negative_laplacian.shape[0])
        stencil = plan.stencil
        with cp.cuda.Device(self.device_id):
            self.cg = FusedConjugateGradient(cp, negative_laplacian, self.device_id, size)
            exterior = sp.csr_matrix(
                (
                    np.asarray(stencil.coefficients, dtype=np.float64),
                    (np.asarray(stencil.rows, dtype=np.int64), np.arange(stencil.size, dtype=np.int64)),
                ),
                shape=(size, max(stencil.size, 1)),
            )
            self.exterior = _upload_csr(exterior, cpsparse)
            boundary = plan.boundary
            if isinstance(boundary, WireBoundaryPlan):
                self.boundary = DeviceWireBoundary(boundary, cp)
            elif isinstance(boundary, SlabBoundaryPlan):
                self.boundary = DeviceSlabBoundary(boundary, cp)
            else:
                raise TypeError(f"no device form of the boundary plan {type(boundary).__name__}")
            self.exterior_count = int(stencil.size)
        self.size = size

    def solve(self, density, settings, initial_potential, *, raise_on_nonconvergence: bool):

        cp = self.cp
        density = np.asarray(density, dtype=np.float64)
        with cp.cuda.Device(self.device_id):
            device_density = cp.asarray(density)
            if self.exterior_count:
                values = self.boundary.values(device_density)
                correction = self.exterior @ values[: self.exterior.shape[1]]
            else:
                values = cp.zeros(0, dtype=cp.float64)
                correction = cp.zeros(self.size, dtype=cp.float64)
            rhs = 8.0 * np.pi * device_density - correction
            if initial_potential is None:
                initial = cp.zeros(self.size, dtype=cp.float64)
            else:
                initial = cp.array(np.asarray(initial_potential, dtype=np.float64), copy=True)
            potential, converged, iterations, matvecs, residual, initial_residual = (
                self.cg.solve(rhs, initial, settings)
            )
            result = ConfinedHartreeResult(
                potential=cp.asnumpy(potential),
                right_hand_side=cp.asnumpy(rhs),
                boundary_values=cp.asnumpy(values),
                converged=converged,
                iterations=iterations,
                matrix_vector_products=matvecs,
                residual_norm=residual,
                initial_residual_norm=initial_residual,
            )
        if not converged and raise_on_nonconvergence:
            raise RuntimeError(
                "wire/slab Hartree conjugate-gradient solve did not converge: "
                f"residual={residual:.3e}, matvecs={matvecs}"
            )
        return result

    @property
    def device_bytes(self) -> int:
        exterior = self.exterior
        return int(
            self.cg.device_bytes
            + exterior.data.nbytes + exterior.indices.nbytes + exterior.indptr.nbytes
            + self.boundary.device_bytes
        )


class DevicePeriodicHartree:
    """The Hartree solve of a 3-D cell on a device: ``Calc_Hartree_G`` or the FD conjugate gradients."""

    def __init__(self, grid, negative_laplacian, device_id: int, *, g_space: bool) -> None:
        cp, cpsparse = _cupy()
        self.cp = cp
        self.device_id = int(device_id)
        self.size = int(grid.size)
        self.g_space = bool(g_space)
        self.shape = tuple(int(value) for value in grid.points_per_axis)
        with cp.cuda.Device(self.device_id):
            if self.g_space:
                from parsec_python.Hartree.gspace import g_squared

                squared = g_squared(grid)
                squared[0, 0, 0] = 1.0
                factor = 8.0 * np.pi / squared
                factor[0, 0, 0] = 0.0
                self.factor = cp.asarray(factor)
                self.cg = None
            else:
                self.factor = None
                self.cg = FusedConjugateGradient(cp, negative_laplacian, self.device_id, self.size)

    @property
    def device_bytes(self) -> int:
        if self.factor is not None:
            return int(self.factor.nbytes)
        return int(self.cg.device_bytes)

    def solve(self, density, settings, initial_potential, *, raise_on_nonconvergence: bool):
        cp = self.cp
        density = np.asarray(density, dtype=np.float64)
        if density.shape != (self.size,):
            raise ValueError("density does not match the active grid")
        if initial_potential is not None and np.shape(initial_potential) != (self.size,):
            raise ValueError("initial Hartree potential does not match the grid")
        with cp.cuda.Device(self.device_id):
            device_density = cp.asarray(density)
            neutral = device_density - cp.mean(device_density)
            rhs = 8.0 * np.pi * neutral
            if self.g_space:
                # Hartree.gspace.solve_hartree_g_space: V(G) = 8 pi n(G)/|G|^2, V(0) = 0.
                transform = cp.fft.fftn(device_density.reshape(self.shape))
                transform *= self.factor
                potential = cp.real(cp.fft.ifftn(transform)).reshape(-1)
                potential = potential - cp.mean(potential)
                result = PeriodicHartreeResult(
                    potential=cp.asnumpy(potential),
                    right_hand_side=cp.asnumpy(rhs),
                    converged=True,
                    iterations=0,
                    matrix_vector_products=0,
                    residual_norm=0.0,
                    initial_residual_norm=_device_scalar(cp.linalg.norm(rhs)),
                )
                return result

            if initial_potential is None:
                initial = cp.zeros(self.size, dtype=cp.float64)
            else:
                initial = cp.array(np.asarray(initial_potential, dtype=np.float64), copy=True)
                initial = initial - cp.mean(initial)
            potential, converged, iterations, matvecs, residual, initial_residual = (
                self.cg.solve(rhs, initial, settings)
            )
            potential = potential - cp.mean(potential)
            result = PeriodicHartreeResult(
                potential=cp.asnumpy(potential),
                right_hand_side=cp.asnumpy(rhs),
                converged=converged,
                iterations=iterations,
                matrix_vector_products=matvecs,
                residual_norm=residual,
                initial_residual_norm=initial_residual,
            )
        if not converged and raise_on_nonconvergence:
            raise RuntimeError(
                "periodic Hartree conjugate-gradient solve did not converge: "
                f"residual={residual:.3e}, matvecs={matvecs}"
            )
        return result


class _SolverCache:
    """The device problems of the last few grids, keyed by the identity of their host objects.

    The entry holds the host objects of its key, so the identities stay valid while it lives.
    """

    def __init__(self, capacity: int = _CACHE_CAPACITY) -> None:
        self.capacity = int(capacity)
        self._entries: OrderedDict[tuple, tuple[tuple, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key_objects: tuple, extra: tuple, build):
        key = tuple(id(item) for item in key_objects) + extra
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry[1]
        solver = build()
        with self._lock:
            self._entries[key] = (key_objects, solver)
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        return solver

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


_CACHE = _SolverCache()


def clear_cache() -> None:
    """Release every cached device problem."""

    _CACHE.clear()


def confined_device_solver(plan, negative_laplacian):
    """The device problem of a wire or slab solve, or ``None`` where the host solves."""

    if hartree_device_mode() == "host":
        return None
    if not isinstance(plan.boundary, (WireBoundaryPlan, SlabBoundaryPlan)):
        return None
    if isinstance(plan.boundary, SlabBoundaryPlan) and getattr(plan.boundary.to_box, "__self__", None) is None:
        return None
    device_id = hartree_device_id()
    if device_id is None:
        return None
    return _CACHE.get(
        (plan, negative_laplacian),
        ("confined", device_id),
        lambda: DeviceConfinedHartree(plan, negative_laplacian, device_id),
    )


def periodic_device_solver(grid, negative_laplacian, settings):
    """The device problem of a 3-D cell's solve, or ``None`` where the host solves."""

    mode = hartree_device_mode()
    if mode == "host":
        return None
    g_space = bool(getattr(settings, "periodic_g_space", False))
    if not g_space:
        # The finite-difference CG of an orthorhombic cell (by default since
        # 2026-10-30; PARSEC_PERIODIC_HARTREE=host keeps the host CG); the
        # general grid's FFT of the stencil symbol stays on the host.
        if getattr(grid, "is_general_periodic", False):
            return None
    device_id = hartree_device_id()
    if device_id is None:
        return None
    return _CACHE.get(
        (grid, negative_laplacian),
        ("g_space" if g_space else "cg", device_id),
        lambda: DevicePeriodicHartree(grid, negative_laplacian, device_id, g_space=g_space),
    )


def solve_confined_on_device(
    density,
    plan,
    negative_laplacian,
    settings,
    initial_potential,
    *,
    raise_on_nonconvergence: bool = True,
):
    """The hook of :func:`parsec_python.Hartree.confined.solve_confined_hartree`."""

    solver = confined_device_solver(plan, negative_laplacian)
    if solver is None:
        return None
    return solver.solve(
        density, settings, initial_potential, raise_on_nonconvergence=raise_on_nonconvergence
    )


def solve_periodic_on_device(
    density,
    grid,
    negative_laplacian,
    settings,
    initial_potential,
    *,
    raise_on_nonconvergence: bool = True,
):
    """The hook of :func:`parsec_python.Hartree.pbc.solve_periodic_hartree`."""

    solver = periodic_device_solver(grid, negative_laplacian, settings)
    if solver is None:
        return None
    return solver.solve(
        density, settings, initial_potential, raise_on_nonconvergence=raise_on_nonconvergence
    )


def describe_periodic_hartree(system, default: str) -> str:
    """The ``hartree_backend`` text of a periodic system's solve (``default`` for the plain FD-CG on the host)."""

    grid = getattr(system, "grid", None)
    plan = getattr(system, "hartree_plan", None)
    settings = getattr(getattr(system, "input", None), "hartree", None)
    try:
        device = hartree_device_mode() != "host" and hartree_device_id() is not None
    except ValueError:
        device = False
    if getattr(grid, "is_confined_periodic", False) and plan is not None:
        kind = "wire" if int(getattr(grid, "periodicity", 0)) == 1 else "slab"
        where = "CuPy float64 on the device" if device else "the host (SciPy CG)"
        return (
            f"{kind} boundary values (hartset_{kind}) and conjugate gradients, "
            f"{where} (Hartree.confined, acceleration/Hartree/periodic_device.py)"
        )
    if getattr(settings, "periodic_g_space", False):
        where = "CuPy FFT on the device" if device else "NumPy FFT on the host"
        return f"reciprocal-space solve of Calc_Hartree_G, {where} (Hartree.gspace)"
    if device and not getattr(grid, "is_general_periodic", False):
        return (
            "periodic conjugate-gradient solve, CuPy float64 on the device "
            f"({MODE_ENV}={hartree_device_mode()}; SCF.pbc.solve_periodic_hartree)"
        )
    return default


def install() -> None:
    """Make these the solvers of the readable wire/slab and periodic Hartree functions."""

    set_confined_hartree_solver(solve_confined_on_device)
    set_periodic_hartree_solver(solve_periodic_on_device)


def uninstall() -> None:
    """Restore the readable solves everywhere (tests)."""

    set_confined_hartree_solver(None)
    set_periodic_hartree_solver(None)


__all__ = [
    "DeviceConfinedHartree",
    "DevicePeriodicHartree",
    "DeviceSlabBoundary",
    "DeviceWireBoundary",
    "FusedConjugateGradient",
    "MODE_ENV",
    "clear_cache",
    "confined_device_solver",
    "describe_periodic_hartree",
    "hartree_device_id",
    "hartree_device_mode",
    "install",
    "periodic_device_solver",
    "solve_confined_on_device",
    "solve_periodic_on_device",
    "uninstall",
]
