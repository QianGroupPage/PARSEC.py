"""Device-resident Hartree solve on the symmetry wedge of the SCF fields.

Where the conjugate gradients of the Hartree solve run on a GPU on the
normalized symmetry wedge that the SCF fields use (one physical value per
orbit, :class:`..SCF.symmetry_fields.SymmetryScalarField`) and the boundary
values are built on the same device, :class:`CuPyResidentHartree` keeps every
array of a solve on that device.  The density goes up as one value per orbit;
the multipoles, the boundary values and the normalized right-hand side

``b = U^T (8 pi rho - A_IB V_B)``,

the chronological starting vector and the CG all run there; the potential
and the right-hand side come back as one value per orbit.  ``U`` has one
column ``e_O / sqrt(m_O)`` per orbit ``O`` of ``m_O`` grid points.  The host
interface of ``driver.py`` instead receives the right-hand side on the host,
extrapolates the starting vector there and uploads both for every solve, and
with a full-grid boundary builder it expands the density to the full grid on
the host and projects the full right-hand side there as well.

A builder of the wedge (``build_reduced_device``) returns ``b`` itself, and
no full-grid map is uploaded: a full-grid field, which the SCF does not
pass, is reduced to the wedge on the host (``reduce_vector``).  A
full-grid builder (``build_device``) receives the density expanded on the
device through ``full_to_wedge`` and its right-hand side is projected there,
summed over the members of each orbit in increasing grid order as
``reduce_vector`` sums them.

The arithmetic is that of the host interface: the same recurrence of the
same CG backend, the same controls and the predictor of
:class:`.native_poisson.NativePoissonSolver` (``x = x_1 + alpha (x_1 - x_2)``
with ``alpha = <d, b - b_1>/<d, d>``, ``d = b_1 - b_2``, clipped to
``[-0.5, 1.5]``), evaluated on the host from the downloaded right-hand side
with ``PARSEC_CUPY_RESIDENT_PREDICTOR=host`` (the default) and on the device
with ``device``.  The potential and the right-hand side are returned as the
orbit averages of their expanded fields (the sum of ``m_O`` equal values
divided by ``m_O``), which is what the host interface forms through
``expand_vector`` and the orbit sums; with the device predictor they are
divided by ``sqrt(m_O)`` directly.  Results differ from the host interface
at round-off only.
"""

from __future__ import annotations

import os
from time import perf_counter

import numpy as np

from ..backends.cupy import require_cupy
from ..SCF.symmetry_fields import SymmetryScalarField
from .native_poisson import NativePoissonResult

# Both kernels are compiled without fused multiply-adds, as the CG is.
_SOURCE = r"""
// out[w] = sum over the members k of orbit w of full[k], divided by
// sqrt(m_w): U^T of a full-grid vector, in increasing grid order.
extern "C" __global__ void reduce_ordered(
    int n, const int* ptr, const int* members, const double* roots,
    const double* full, double* out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    double sum = 0;
    for (int k = ptr[row]; k < ptr[row + 1]; ++k) sum += full[members[k]];
    out[row] = sum / roots[row];
}

// The orbit averages of the expanded potential (out[row]) and right-hand
// side (out[n + row]): expand_vector gives every member normalized/sqrt(m),
// and the m equal values are summed and divided by m without a full array.
extern "C" __global__ void physical_average(
    int n, const int* ptr, const double* roots, const double* normalized,
    const double* rhs, double* out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    int multiplicity = ptr[row + 1] - ptr[row];
    double v = normalized[row] / roots[row], b = rhs[row] / roots[row];
    double sv = 0, sb = 0;
    for (int k = 0; k < multiplicity; ++k) {
        sv += v;
        sb += b;
    }
    out[row] = sv / multiplicity;
    out[n + row] = sb / multiplicity;
}
"""
_BLOCK = 256
_CONTROLS = ("relative_tolerance", "absolute_tolerance", "max_iterations")


def resident_predictor_setting() -> str:
    """``PARSEC_CUPY_RESIDENT_PREDICTOR``: ``host`` (the default) or ``device``."""

    predictor = os.environ.get("PARSEC_CUPY_RESIDENT_PREDICTOR", "host").strip().lower()
    if predictor not in {"host", "device"}:
        raise ValueError("PARSEC_CUPY_RESIDENT_PREDICTOR must be host or device")
    return predictor


def _chronological_guess_requested() -> bool:
    return os.environ.get("PARSEC_HARTREE_CHRONOLOGICAL_GUESS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


class CuPyResidentHartree:
    """Boundary, projection, starting vector and CG of each solve on one device.

    ``builder`` is a GPU boundary builder of ``Hartree/cupy_boundary.py`` on
    the device of ``backend``, the prepared CG of the normalized wedge of
    ``reduction`` (``CuPyPreparedConjugateGradientBackend``); ``settings``
    are the :class:`parsec_python.models.HartreeSettings` of the solves.
    The grid must be indexable by ``int32``.

    :meth:`solve` has the contract of the solver of the host interface: a
    density and a starting potential as :class:`SymmetryScalarField` of this
    wedge (or full-grid arrays), the CG controls ``relative_tolerance``,
    ``absolute_tolerance``, ``max_iterations`` and
    ``raise_on_nonconvergence``, and a :class:`HartreeResult` with fields of
    one value per orbit.  ``rhs_seconds`` and ``solve_seconds`` are the
    seconds of the last solve, the multipoles and right-hand side and the
    rest; ``history`` holds the last two pairs of right-hand side and
    solution of the predictor (host arrays with the host predictor).
    """

    def __init__(self, builder, reduction, backend, settings):
        cp, _ = require_cupy()
        self.cp = cp
        self.builder = builder
        self.reduction = reduction
        self.backend = backend
        self.settings = settings
        self.history = []
        self.predictor = resident_predictor_setting()
        if reduction.full_size > np.iinfo(np.int32).max:
            raise ValueError("the resident Hartree chain needs a grid of int32 indices")
        # A builder of the wedge returns the normalized wedge right-hand side
        # itself and needs neither full-grid map; a full-grid builder takes
        # both (the expansion of the density and the projection of its
        # right-hand side).
        self.wedge_builder = hasattr(builder, "build_reduced_device")
        self.map = None
        self.members = None
        with cp.cuda.Device(builder.device_id):
            self.roots = cp.asarray(np.sqrt(reduction.multiplicities))
            self.ptr = cp.asarray(
                np.r_[0, np.cumsum(reduction.multiplicities)], dtype=cp.int32
            )
            if not self.wedge_builder:
                self._upload_full_maps()
            options = ("--std=c++11", "--fmad=false")
            self.kernel = cp.RawKernel(_SOURCE, "reduce_ordered", options=options)
            self.kernel.compile()
            self.export_kernel = cp.RawKernel(_SOURCE, "physical_average", options=options)
            self.export_kernel.compile()
        self.rhs_seconds = 0.0
        self.solve_seconds = 0.0

    @property
    def device_id(self) -> int:
        """The CUDA device of the chain: that of its boundary builder and CG."""

        return int(self.builder.device_id)

    @property
    def device_bytes(self) -> int:
        """Device bytes the chain holds and works in, for the memory rules.

        The square roots of the multiplicities and the orbit pointer, the
        two full-grid maps of a full-grid builder, the history of the
        device predictor, and the working arrays of one solve: six wedge
        vectors (density, right-hand side, starting vector, solution and
        the two returned fields), four more for the device predictor, and
        the expanded density of a full-grid builder.  The arrays of the
        builder and of the CG are theirs (``device_storage_bytes``,
        ``device_bytes``).
        """

        if self.roots is None:
            return 0
        wedge = int(self.reduction.wedge_size)
        held = int(self.roots.nbytes) + int(self.ptr.nbytes)
        if self.map is not None:
            held += int(self.map.nbytes) + int(self.members.nbytes)
        working = 6 * wedge
        if self.predictor == "device":
            held += 4 * 8 * wedge
            working += 4 * wedge
        if not self.wedge_builder:
            working += int(self.reduction.full_size)
        return held + 8 * working

    def _upload_full_maps(self):
        cp = self.cp
        reduction = self.reduction
        self.map = cp.asarray(reduction.full_to_wedge, dtype=cp.int32)
        self.members = cp.asarray(
            np.argsort(reduction.full_to_wedge, kind="stable"), dtype=cp.int32
        )

    def project(self, full):
        """``U^T full`` of a full-grid device vector, on the wedge."""

        cp = self.cp
        if self.members is None:
            self._upload_full_maps()
        out = cp.empty(self.reduction.wedge_size, cp.float64)
        self.kernel(
            ((out.size + _BLOCK - 1) // _BLOCK,),
            (_BLOCK,),
            (np.int32(out.size), self.ptr, self.members, self.roots, full, out),
        )
        return out

    def _device_density(self, density):
        """The density in the layout the builder takes, on the device."""

        cp = self.cp
        if isinstance(density, SymmetryScalarField):
            if density.reduction is not self.reduction:
                raise ValueError("the resident Hartree density uses another wedge")
            values = cp.asarray(density.values)
            # A full-grid builder takes the physical value at every point.
            return values if self.wedge_builder else values[self.map]
        if not self.wedge_builder:
            return cp.asarray(density)
        # One physical value per orbit, the orbit average: summed on the host
        # in the order of the device projection, so that no full-grid map is
        # kept on the device for a field that rarely comes.
        return cp.asarray(self.reduction.reduce_vector(np.asarray(density))) / self.roots

    def _starting_vector(self, rhs, initial_potential):
        """The normalized starting vector of the given potential, on the device."""

        cp = self.cp
        if initial_potential is None:
            return cp.zeros_like(rhs)
        if isinstance(initial_potential, SymmetryScalarField):
            if initial_potential.reduction is not self.reduction:
                raise ValueError("the resident Hartree starting potential uses another wedge")
            return cp.asarray(initial_potential.values) * self.roots
        if self.wedge_builder:
            return cp.asarray(self.reduction.reduce_vector(np.asarray(initial_potential)))
        return self.project(cp.asarray(initial_potential))

    def _predicted(self, rhs, host_rhs):
        """The chronological starting vector, ``None`` where it does not apply."""

        if len(self.history) != 2 or not _chronological_guess_requested():
            return None
        cp = self.cp
        (old_rhs, old_x), (prev_rhs, prev_x) = self.history
        on_host = self.predictor == "host"
        xp = np if on_host else cp
        direction = prev_rhs - old_rhs
        denominator = xp.dot(direction, direction)
        host_denominator = float(denominator) if on_host else float(denominator.get())
        if not (np.isfinite(host_denominator) and host_denominator > 0):
            return None
        new_rhs = host_rhs if on_host else rhs
        alpha = xp.clip(xp.dot(direction, new_rhs - prev_rhs) / denominator, -0.5, 1.5)
        predicted = prev_x + alpha * (prev_x - old_x)
        # The guard of NativePoissonSolver: no prediction that is not finite.
        return predicted if bool(xp.all(xp.isfinite(predicted))) else None

    def solve(self, density, initial_potential=None, **controls):
        """Solve for the Hartree potential of ``density`` (see the class)."""

        cp = self.cp
        reduction = self.reduction
        settings = {
            name: controls.pop(name, getattr(self.settings, name)) for name in _CONTROLS
        }
        raise_on_nonconvergence = controls.pop("raise_on_nonconvergence", True)
        if controls:
            raise TypeError(f"unsupported resident Hartree controls: {tuple(controls)}")
        with cp.cuda.Device(self.builder.device_id):
            started = perf_counter()
            device_density = self._device_density(density)
            if self.wedge_builder:
                # The builder lends its buffer, and the predictor keeps the
                # right-hand sides of two solves.
                rhs, boundary = self.builder.build_reduced_device(device_density)
                rhs = rhs.copy()
            else:
                full_rhs, boundary = self.builder.build_device(device_density)
                rhs = self.project(full_rhs)
            cp.cuda.get_current_stream().synchronize()
            self.rhs_seconds = perf_counter() - started

            started = perf_counter()
            host_rhs = cp.asnumpy(rhs) if self.predictor == "host" else None
            initial = self._starting_vector(rhs, initial_potential)
            predicted = self._predicted(rhs, host_rhs)
            if predicted is not None:
                initial = predicted
            payload = self.backend.solve(rhs, initial, device_result=True, **settings)
            if not payload["converged"] and raise_on_nonconvergence:
                raise RuntimeError("the resident Hartree CG did not converge")
            solution = payload.pop("solution")
            self.history.append(
                (host_rhs, cp.asnumpy(solution))
                if self.predictor == "host"
                else (rhs, solution)
            )
            self.history = self.history[-2:]
            # The SCF algebra on the host receives physical values per orbit.
            if self.predictor == "host":
                wedge = reduction.wedge_size
                physical = cp.empty((2, wedge), cp.float64)
                self.export_kernel(
                    ((wedge + _BLOCK - 1) // _BLOCK,),
                    (_BLOCK,),
                    (np.int32(wedge), self.ptr, self.roots, solution, rhs, physical),
                )
                host = cp.asnumpy(physical)
            else:
                host = cp.asnumpy(cp.stack((solution / self.roots, rhs / self.roots)))
            self.solve_seconds = perf_counter() - started
        result = NativePoissonResult(
            potential=SymmetryScalarField(reduction, host[0]),
            right_hand_side=SymmetryScalarField(reduction, host[1]),
            **payload,
        )
        return result.as_hartree_result(boundary)

    def host_history(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The predictor history as host arrays of the normalized wedge."""

        cp = self.cp
        return [
            (np.asarray(cp.asnumpy(rhs)), np.asarray(cp.asnumpy(solution)))
            for rhs, solution in self.history
        ]

    def release(self) -> None:
        """Drop the device arrays of the chain and its references to the builder and CG."""

        self.history = []
        self.map = self.members = None
        self.roots = self.ptr = None
        self.builder = self.backend = None
