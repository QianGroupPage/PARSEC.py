"""Bounded FP64 atom/grid setup; atom order and radial rules match native.

Each CUDA thread owns one grid point and visits atoms in input order. There
are no atomics, atom-by-grid intermediates, or truncation of Coulomb tails.
Temporary CUDA allocations bypass the orbital pool and are released before
SCF. Independent grid slabs are evaluated on every device of an MPI rank
that solves symmetry sectors and on one device otherwise, unless
``PARSEC_IONIC_GPU_COUNT`` names a count (see :func:`ionic_device_ids`).

A field is totally symmetric under the symmetry group of the structure. The
builders can sum it at one point per symmetry orbit and give every point of
the orbit that value (:meth:`CupyIonicBuilders.sum_on_orbits`); whoever asks
for that has to know that the images of atoms and grid points are exact
(``Symmetry/exact_images.py``).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os

import numpy as np

from .native_ionic import NativeIonicBuilders, _density_table, _spline_payload, _values64
from ..backends.cupy import require_cupy
from ..backends.cupy_compile import compile_cupy_raw


_SOURCE = r'''
__device__ int interval(const double* x, int n, double q) {
    int lo=0, hi=n;
    while (lo < hi) { int mid=(lo+hi)/2; if (q < x[mid]) hi=mid; else lo=mid+1; }
    return lo <= 0 ? 0 : (lo >= n ? n-2 : lo-1);
}
__device__ double spline(const double* x, const double* v, const double* s,
                         int n, double q) {
    int a=interval(x,n,q), b=a+1;
    double step=x[b]-x[a], left=(x[b]-q)/step, right=(q-x[a])/step;
    return left*v[a]+right*v[b]+
        ((left*left*left-left)*s[a]+(right*right*right-right)*s[b])*step*step/6.0;
}
extern "C" __global__ void ionic_sum(
    const double* xyz, long long n, const double* positions, const int* types,
    int natoms, const double* tables, const int* descriptors,
    const double* charges, int local, double* output) {
    long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (row>=n) return;
    double gx=xyz[3*row], gy=xyz[3*row+1], gz=xyz[3*row+2], total=0.0;
    for(int atom=0; atom<natoms; ++atom) {
        double x=gx-positions[3*atom], y=gy-positions[3*atom+1], z=gz-positions[3*atom+2];
        double distance=sqrt(x*x+y*y+z*z);
        int species=types[atom];
        const int* d=descriptors+7*species;
        const double* r=tables+d[0];
        const double* v=tables+d[2];
        double value=0.0;
        if(distance>=r[d[1]-2]) {
            if(local) value=-2.0*charges[species]/distance;
        } else if(d[6]) {
            value=spline(tables+d[3],tables+d[4],tables+d[5],d[6],distance);
        } else if(distance<=r[0]) {
            value=v[0];
        } else {
            int a=interval(r,d[1],distance), b=a+1;
            double fraction=(distance-r[a])/(r[b]-r[a]);
            double va=local ? r[a]*v[a] : v[a], vb=local ? r[b]*v[b] : v[b];
            value=va+fraction*(vb-va);
            if(local) value/=distance;
        }
        total+=value;
    }
    output[row]=total;
}
'''


@lru_cache(maxsize=1)
def _kernel():
    cp, _ = require_cupy()
    return cp.RawKernel(_SOURCE, "ionic_sum", options=("--std=c++17", "--fmad=false"))


def ionic_gpu_count_setting() -> str:
    """Return ``PARSEC_IONIC_GPU_COUNT`` as given: ``auto``, the default, or a number."""

    return os.environ.get("PARSEC_IONIC_GPU_COUNT", "auto").strip().lower() or "auto"


def ionic_sum_grid_setting() -> str:
    """Return ``PARSEC_IONIC_SUM_GRID``: ``auto``, the default, or ``full``.

    ``auto`` lets ionic fields that are built on their own thread be summed
    on the symmetry wedge, one point per orbit, where the images of the
    atoms and of the grid points under the detected group are exact in
    every bit; anywhere else, and with ``full``, every grid point is summed
    as before.  A wedge value is the full-grid value at the representative
    point of its orbit, bit for bit; the other points of the orbit take it
    in place of their own sum, which adds the same terms in another order
    of the atoms: the fields change by the round-off of one sum.
    """

    value = os.environ.get("PARSEC_IONIC_SUM_GRID", "auto").strip().lower() or "auto"
    if value not in {"auto", "full"}:
        raise ValueError("PARSEC_IONIC_SUM_GRID must be auto or full")
    return value


def ionic_device_ids(available, grid_size, *, sector_rank=False):
    """Return the devices among ``available`` that share one ionic sum.

    ``available`` are the devices of the process in the order of
    ``PARSEC_CUPY_DEVICES``.  A number in ``PARSEC_IONIC_GPU_COUNT`` takes
    the first that many; 1 is the former default.

    ``auto``, the default, takes a device only where the calculation is
    known to use it.  The sums may run before the symmetry of the structure
    is detected (in line, or on their thread with
    ``PARSEC_IONIC_SUM_GRID=full``), and this rule does not wait for it, so
    that is known of all the devices only for ``sector_rank``,
    an MPI rank that solves symmetry sectors: its sector groups cover its
    devices, and the MPI runner creates a CUDA context on each, before the
    preparation or on a thread beside it; a sum that reaches a device first
    waits for that context or creates it.
    Any other process takes the first device.  It may go on to solve without
    symmetry, on one device, and a sum on the others would create CUDA
    contexts there (426 MiB each on an A100) that nothing uses afterwards.

    A grid point is summed by one thread of one device whichever is chosen,
    so the fields do not depend on the count.
    """

    available = tuple(int(device) for device in available)
    raw = ionic_gpu_count_setting()
    if raw == "auto":
        count = len(available) if sector_rank else 1
    else:
        try:
            count = int(raw)
        except ValueError as error:
            raise ValueError(
                "PARSEC_IONIC_GPU_COUNT must be auto or a positive integer"
            ) from error
        if count < 1:
            raise ValueError("PARSEC_IONIC_GPU_COUNT must be positive")
    devices = available[:min(count, len(available), max(int(grid_size), 1))]
    if not devices:
        raise ValueError("GPU ionic setup requires at least one selected device")
    return devices


class CupyIonicBuilders(NativeIonicBuilders):
    """GPU local/density fields and native support-localized KB projectors.

    ``sector_rank`` says that the process is an MPI rank that solves symmetry
    sectors (see :func:`ionic_device_ids`).  ``device_ids`` are the devices
    of the latest sum, empty before the first one.  ``current_device`` is the
    device the process setting ``current`` stands for: that of the thread
    that sums, unless :meth:`keep_current_device` has kept another.
    ``summed_points`` is the number of points the latest sum ran on: the
    size of the grid, or the number of its orbits after
    :meth:`sum_on_orbits`.
    """

    def __init__(self, *, sector_rank: bool = False) -> None:
        super().__init__()
        self.sector_rank = bool(sector_rank)
        self.device_ids: tuple[int, ...] = ()
        self.current_device: int | None = None
        self.summed_points = 0
        # The grid whose sums run on one point per orbit, the coordinates of
        # those points and the orbit of every grid point; None: every point.
        self._orbits = None

    def sum_on_orbits(self, grid, reduction, coordinates=None) -> None:
        """Sum the fields of ``grid`` at one point per orbit of ``reduction`` from now on.

        The sums run at the representative rows of the reduction, in their
        order, and every row of the grid is given the value of its orbit:
        the arrays returned keep the size and the layout of the full grid.
        ``coordinates`` are the coordinates of the representative rows
        where the caller has gathered them.

        The value at a representative row is the one a sum over the whole
        grid gives that row, bit for bit: the kernel reads the coordinates
        of one point and the atoms in their order.  That the other points of
        an orbit may take it is the caller's to know.  Sums of another grid
        stay on all its points.
        """

        rows = np.asarray(reduction.representative_rows)
        orbit = np.asarray(reduction.full_to_wedge)
        if orbit.shape != (grid.size,):
            raise ValueError("the symmetry reduction does not match the grid")
        if coordinates is None:
            coordinates = grid.coordinates[rows]
        coordinates = np.ascontiguousarray(coordinates, dtype=np.float64)
        if coordinates.shape != (rows.size, 3):
            raise ValueError("one coordinate triple per symmetry orbit is required")
        self._orbits = (grid, coordinates, orbit)

    def sum_on_full_grid(self) -> None:
        """Sum every grid point again, as before :meth:`sum_on_orbits`."""

        self._orbits = None

    def keep_current_device(self) -> None:
        """Keep the current CUDA device of the calling thread for later sums.

        The current device belongs to a thread and a new thread starts with
        device 0.  A caller that hands the sums to another thread calls this
        first, so that they read ``PARSEC_CUPY_DEVICES`` as its eigensolver
        will.
        """

        cp, _ = require_cupy()
        self.current_device = int(cp.cuda.Device().id)

    def _sum(self, grid, atoms, potentials, specifications, *, local, core=False):
        cp, _ = require_cupy()
        active = [a for a in atoms if not core or potentials[a.symbol].has_nonlinear_core_correction]
        if not active:
            return np.zeros(grid.size, dtype=np.float64)
        symbols = list(dict.fromkeys(a.symbol for a in active))
        parts, descriptors, charges = [], [], []
        cursor = 0

        def append(values):
            nonlocal cursor
            offset = cursor
            part = _values64(values)
            parts.append(part)
            cursor += part.size
            return offset

        for symbol in symbols:
            potential, spec = potentials[symbol], specifications[symbol]
            if local:
                values = _values64(potential.channel_potentials[spec.local_angular_momentum])
                use_spline = spec.use_spline
            else:
                values, use_spline = _density_table(potential, spec, core=core)
            knots, spl_values, second = _spline_payload(potential, values, grid, use_spline)
            descriptors.append([
                append(potential.radii), potential.radii.size, append(values),
                append(knots), append(spl_values), append(second), knots.size,
            ])
            charges.append(potential.ionic_charge)
        tables = np.concatenate(parts)
        descriptors = np.asarray(descriptors, dtype=np.int32)
        positions = _values64([a.position for a in active])
        types = np.asarray([symbols.index(a.symbol) for a in active], dtype=np.int32)
        charges = _values64(charges)
        # The devices the eigensolver of this process will use, read the same way.
        from ..Eigensolvers.symmetry import selected_device_ids

        current = self.current_device
        if current is None:
            current = int(cp.cuda.Device().id)
        # Every point of the grid, or the representative point of every
        # symmetry orbit of this very grid (sum_on_orbits).
        orbits = self._orbits
        if orbits is not None and orbits[0] is not grid:
            orbits = None
        points = grid.coordinates if orbits is None else orbits[1]
        count = grid.size if orbits is None else int(points.shape[0])
        devices = ionic_device_ids(
            selected_device_ids(current),
            count,
            sector_rank=self.sector_rank,
        )
        self.device_ids = devices
        self.summed_points = count
        output = np.empty(count, dtype=np.float64)

        def slab(item):
            slot, device = item
            start = count * slot // len(devices)
            stop = count * (slot + 1) // len(devices)
            def direct_allocate(size):
                return cp.cuda.MemoryPointer(cp.cuda.Memory(size), 0)

            with cp.cuda.Device(device), cp.cuda.using_allocator(direct_allocate):
                xyz = cp.asarray(points[start:stop], order="C")
                inputs = [cp.asarray(a) for a in (positions, types, tables, descriptors, charges)]
                result = cp.empty(stop-start, dtype=cp.float64)
                _kernel()(((stop-start+127)//128,), (128,), (
                    xyz, np.int64(stop-start), inputs[0], inputs[1], np.int32(len(active)),
                    inputs[2], inputs[3], inputs[4], np.int32(local), result,
                ))
                result.get(out=output[start:stop])
                # Free on the owning device, before returning from its context.
                del xyz, inputs, result

        if count:
            if len(devices) == 1:
                slab((0, devices[0]))
            else:
                # Compile here once, so that the device threads find the
                # kernel in CuPy's cache instead of compiling it side by side.
                with cp.cuda.Device(devices[0]):
                    compile_cupy_raw(_kernel())
                with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                    list(executor.map(slab, enumerate(devices)))
        if orbits is None:
            return output
        # Every row of the grid takes the value of its orbit.
        return output.take(orbits[2])

    def build_local_ionic_potential(self, grid, atoms, potentials, specifications):
        return self._sum(grid, atoms, potentials, specifications, local=True)

    def superpose_atomic_density(self, grid, atoms, potentials, specifications, *, core=False):
        return self._sum(grid, atoms, potentials, specifications, local=False, core=core)
