"""The direct Coulomb boundary of the Hartree problem, summed on a GPU.

A cluster domain that is no sphere (``Cluster_Domain_Shape: ellipsoid`` or
``cylinder``, ``grid_partition.f90``'s ``outside_domain``) takes the boundary
values of its Hartree potential from the direct sum over the density on the
grid, as the reference solver does
(:class:`parsec_python.Hartree.DirectCoulombBoundary`, PARSEC's
``Full_Hartree``):

``V_B(R_b) = 2 sum_s rho_s h^3 / |R_b - r_s|``   [Ry]

at every exterior point ``R_b`` that a stencil of an active row reaches, and
the right-hand side of the Poisson problem is

``b = 8 pi rho - A_IB V_B``

(:func:`parsec_python.Laplacian.apply_negative_laplacian_boundary`).  The
sum costs ``O(exterior points x grid points)`` per solve, which the host
evaluates in chunks of sixteen exterior points; here one GPU thread sums
the sources of one exterior point, in tiles of the sources held in shared
memory, with a compensated (Kahan) sum in FP64.  The exterior points, their
rows and coefficients are found once from the stencil walk of the reference
(``neighbor_shells``), and the correction is subtracted on the host in the
order of that walk, so that the right-hand side differs from the
reference's only by the rounding of the sums ``V_B`` (relative 1e-15).
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from parsec_python.Hartree import DirectCoulombBoundary
from parsec_python.Laplacian import neighbor_shells, second_derivative_coefficients

from ..backends.cupy import require_cupy

_SOURCE = r"""
extern "C" __global__ void direct_boundary(
    const long long count,
    const double* __restrict__ points,
    const long long sources,
    const double* __restrict__ coordinates,
    const double* __restrict__ weights,
    double* __restrict__ values)
{
    __shared__ double sx[256];
    __shared__ double sy[256];
    __shared__ double sz[256];
    __shared__ double sw[256];
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double px = 0.0, py = 0.0, pz = 0.0;
    if (row < count) {
        px = points[3 * row];
        py = points[3 * row + 1];
        pz = points[3 * row + 2];
    }
    double sum = 0.0, carry = 0.0;
    for (long long base = 0; base < sources; base += 256) {
        const long long source = base + threadIdx.x;
        if (source < sources) {
            sx[threadIdx.x] = coordinates[3 * source];
            sy[threadIdx.x] = coordinates[3 * source + 1];
            sz[threadIdx.x] = coordinates[3 * source + 2];
            sw[threadIdx.x] = weights[source];
        }
        __syncthreads();
        const long long remaining = sources - base;
        const int tile = remaining < 256 ? (int)remaining : 256;
        if (row < count) {
            for (int k = 0; k < tile; ++k) {
                const double dx = px - sx[k];
                const double dy = py - sy[k];
                const double dz = pz - sz[k];
                const double term = sw[k] / sqrt(dx * dx + dy * dy + dz * dz);
                const double y = term - carry;
                const double t = sum + y;
                carry = (t - sum) - y;
                sum = t;
            }
        }
        __syncthreads();
    }
    if (row < count) {
        values[row] = 2.0 * sum;
    }
}
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    cp, _ = require_cupy()
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(
            _SOURCE, "direct_boundary", options=("--std=c++11", "--fmad=false")
        )
    return _KERNEL


class CuPyDirectCoulombBoundary:
    """The direct boundary of one non-spherical cluster grid, on one device.

    ``device_id`` is the device of the sums (``None``: the current one at
    construction).  Call :meth:`build` with a full-grid density.
    """

    def __init__(self, grid, *, device_id: int | None = None) -> None:
        cp, _ = require_cupy()
        started = perf_counter()
        self.grid = grid
        self.device_id = int(cp.cuda.Device().id if device_id is None else device_id)
        coefficients = second_derivative_coefficients(grid.settings.expansion_order)
        width = grid.settings.stencil_half_width
        inverse_spacing_squared = 1.0 / grid.spacing**2
        shells = []
        exterior = []
        for _axis, signed_shell, neighbor_rows, integer_points in neighbor_shells(grid):
            missing = np.flatnonzero(neighbor_rows < 0)
            if not missing.size:
                continue
            shells.append(
                (missing, -coefficients[width + abs(signed_shell)] * inverse_spacing_squared)
            )
            exterior.append(integer_points[missing])
        integer = np.concatenate(exterior) if exterior else np.zeros((0, 3), dtype=np.int64)
        unique, inverse = np.unique(integer, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        offsets = np.cumsum([0] + [rows.size for rows, _ in shells])
        self.shells = tuple(
            (rows, float(coefficient), inverse[offsets[index] : offsets[index + 1]])
            for index, (rows, coefficient) in enumerate(shells)
        )
        self.exterior_points = grid.physical_coordinates(unique)
        with cp.cuda.Device(self.device_id):
            self._points = cp.asarray(np.ascontiguousarray(self.exterior_points, dtype=np.float64))
            self._sources = cp.asarray(np.ascontiguousarray(grid.coordinates, dtype=np.float64))
        self.setup_seconds = perf_counter() - started
        self.calls = 0
        self.sum_seconds = 0.0

    @property
    def exterior_point_count(self) -> int:
        return int(self.exterior_points.shape[0])

    def boundary_values(self, weights: np.ndarray) -> np.ndarray:
        """``2 sum_s w_s / |R_b - r_s|`` at the exterior points (host array, Ry)."""

        cp, _ = require_cupy()
        started = perf_counter()
        count = self.exterior_point_count
        with cp.cuda.Device(self.device_id):
            device_weights = cp.asarray(np.ascontiguousarray(weights, dtype=np.float64))
            values = cp.empty(count, dtype=cp.float64)
            if count:
                _kernel()(
                    ((count + 255) // 256,),
                    (256,),
                    (
                        np.int64(count),
                        self._points,
                        np.int64(self._sources.shape[0]),
                        self._sources,
                        device_weights,
                        values,
                    ),
                )
            host = cp.asnumpy(values)
        self.calls += 1
        self.sum_seconds += perf_counter() - started
        return host

    def build(self, density: np.ndarray, settings, boundary_tail=None):
        """The right-hand side ``8 pi rho - A_IB V_B`` and its boundary object.

        The counterpart of :func:`.poisson.build_hartree_problem` for the
        direct method; the boundary object is the reference's, which
        evaluates nothing unless it is asked for values.
        """

        if boundary_tail is not None:
            raise ValueError(
                "the atomic tail complements a multipole boundary, not a direct sum"
            )
        density = np.asarray(density, dtype=np.float64)
        if density.shape != (self.grid.size,):
            raise ValueError("density does not match the active grid")
        weights = density * self.grid.volume_element
        values = self.boundary_values(weights)
        right_hand_side = 8.0 * np.pi * density
        for rows, coefficient, exterior in self.shells:
            right_hand_side[rows] -= coefficient * values[exterior]
        boundary = DirectCoulombBoundary(
            source_coordinates=self.grid.coordinates,
            source_weights=weights,
            chunk_size=settings.direct_chunk_size,
        )
        return right_hand_side, boundary


def direct_boundary_device_setting() -> str:
    """``PARSEC_HARTREE_DIRECT_DEVICE``: ``auto`` (default), ``on`` or ``off``.

    ``auto`` sums the direct boundary of an ellipsoid or a cylinder on the
    GPU and leaves a box on the host sum it always had (its results stay
    those of earlier runs to the last bit); ``on`` takes a box to the GPU as
    well; ``off`` keeps every shape on the host, the control of the device
    sum.  ``1``/``0`` (and true/false, yes/no) stand for on/off.
    """

    import os

    value = os.environ.get("PARSEC_HARTREE_DIRECT_DEVICE", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return "on"
    if value in {"0", "false", "no", "off"}:
        return "off"
    if value != "auto":
        raise ValueError("PARSEC_HARTREE_DIRECT_DEVICE must be auto, on or off")
    return "auto"


def direct_boundary_for(grid, settings, *, selected: str, device_id: int | None = None):
    """The device direct boundary of ``grid``, or ``None`` where it has none.

    A CuPy calculation whose boundary is the direct sum (``auto`` for a
    shape that is no sphere, or ``direct``) gets one for an ellipsoid or a
    cylinder, and for a box where ``PARSEC_HARTREE_DIRECT_DEVICE=on``.
    """

    setting = direct_boundary_device_setting()
    shapes = {"ellipsoid", "cylinder"} | ({"box"} if setting == "on" else set())
    if (
        setting == "off"
        or selected != "cupy"
        or grid.settings.domain_shape not in shapes
        or settings.boundary_method not in ("auto", "direct")
    ):
        return None
    return CuPyDirectCoulombBoundary(grid, device_id=device_id)


__all__ = [
    "CuPyDirectCoulombBoundary",
    "direct_boundary_device_setting",
    "direct_boundary_for",
]
