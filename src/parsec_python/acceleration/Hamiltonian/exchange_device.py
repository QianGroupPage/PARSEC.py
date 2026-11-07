"""Exact exchange of a hybrid functional on the devices of a CuPy calculation.

The physics is that of :mod:`parsec_python.V_xc.exchange_operator` (units Ry
and bohr, orbitals ``q = sqrt(h^3) psi``, occupations ``f`` in ``[0, 1]`` per
spin, spin weight ``w`` 2 for an unpolarized set and 1 per channel):

``W = V_x Psi``,  ``(V_x q)_k = - sum_(j occ) f_j q_j v[q_j q_k / h^3]``,
``M = -Psi^T W = L L^T``,  ``xi = W L^-T``,  ``a_x V_ACE = -a_x xi xi^T``,
``E_x = (w/2) sum_i f_i <q_i|W_i>``

(adaptively compressed exchange: Lin, J. Chem. Theory Comput. 12, 2242
(2016); its use in SPARC: Jing, Suryanarayana, arXiv:2406.16998, Sec. 2.2,
and the batched pair solves of Jing, Sharma, Pask, Suryanarayana,
arXiv:2501.16572, Sec. 3, whose principles are followed here without any
SPARC code; the signs and the triangular factor are those of
:mod:`parsec_python.V_xc.exchange_operator`).  What this module adds is where
the arrays are.

Orbitals.  The eigensolvers of an accelerated calculation keep the orbitals
of a spin channel either as one full-grid block on one device, or in the
symmetry sectors of the exact reflection group, each sector on its device
(:class:`parsec_python.acceleration.Eigensolvers.symmetry.CuPySymmetryOrbitals`),
or on the host where a sector was spilled.  An orbital of sector ``G`` is
``q = U_G c`` with the orthonormal expansion ``U_G`` (phase times
``1/sqrt(|O|)`` on the points of each orbit, :class:`SectorExpansion`).  The
occupied density matrix of each channel is invariant under the group, so
``V_x`` commutes with it: ``V_x U_G c`` lies in sector ``G`` again, ``M`` is
block diagonal over the sectors, and the compressed operator of each
sector is ``-xi_G xi_G^T`` with ``xi_G = U_G^T xi`` (sector rows, the
sector's own columns).  ``W_G = U_G^T W`` and ``xi_G`` therefore take the
memory of the sector's orbitals on the sector's device, nothing more.

Pair densities.  A pair density ``q_j q_k`` of two orbitals of sectors ``A``
and ``B`` lies in sector ``A B`` and needs the full grid for the
Kronecker-product Poisson solver
(:class:`parsec_python.acceleration.Hartree.kronecker_device.DeviceKroneckerPoisson`).
The occupied columns of every sector are cut into blocks of
``Exx_Batch_Size`` columns; a tile is a pair of blocks (each unordered pair
once, a block with itself for its pairs ``i <= j``, an empty target block
with every occupied block).  Tiles go to the devices of the process, the
most expensive first, each to the least loaded device; a device unfolds the
two blocks of a tile to the full grid, solves its pair densities
``Exx_Batch_Size`` at a time, accumulates both blocks' parts of ``W`` on the
full grid and folds them back into their sectors.  The folded parts are
added to ``W_G`` on the device of the sector, with a lock per sector.  Every
pair is solved once, wherever its two orbitals are.

Nothing here prints or opens files.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import math
from threading import Lock
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from parsec_python.V_xc.exact_exchange import OCCUPATION_THRESHOLD


# ---------------------------------------------------------------------------
# Orbitals of one channel, sector by sector.
# ---------------------------------------------------------------------------


# A persistent stream per device for gathering the columns of a basis that
# another device reads (CuPy's pool keeps the blocks freed on a stream for
# that stream: they are returned after every build, release_gather_blocks).
_GATHER_STREAMS: dict[int, Any] = {}
# Bytes of one chunk of a host basis in the products of the energies and of M.
_PRODUCT_CHUNK_BYTES = 256 * 1024 * 1024
_GATHER_LOCK = Lock()


def _gather_stream(cp, device: int):
    """The gather stream of ``device``; the caller holds ``_GATHER_LOCK``."""

    stream = _GATHER_STREAMS.get(int(device))
    if stream is None:
        with cp.cuda.Device(int(device)):
            stream = cp.cuda.Stream(non_blocking=True)
        _GATHER_STREAMS[int(device)] = stream
    return stream


def release_gather_blocks(cp) -> None:
    """Return the pool blocks of the gather streams to their devices."""

    with _GATHER_LOCK:
        streams = list(_GATHER_STREAMS.items())
    for device, stream in streams:
        with cp.cuda.Device(device):
            stream.synchronize()
            cp.get_default_memory_pool().free_all_blocks(stream=stream)


class SectorExpansion:
    """``U_G`` of one symmetry sector: its rows on the full grid and back.

    ``point_rows[p]`` is the sector row of the orbit of full-grid point
    ``p`` (``-1`` where the sector's character rejects the orbit) and
    ``point_weights[p]`` the phase of ``p`` times ``1/sqrt(|O|)`` (zero where
    rejected): ``(U_G c)[p] = point_weights[p] c[point_rows[p]]``.
    ``orbit_points[r]`` lists the full-grid points of row ``r`` (padded with
    ``0`` and weight ``0``): ``(U_G^T q)[r] = sum_k orbit_weights[r, k]
    q[orbit_points[r, k]]``, a fixed-order sum.  Device copies are made per
    device on first use.
    """

    def __init__(
        self,
        full_to_wedge: np.ndarray,
        phases: np.ndarray,
        sector_orbits: np.ndarray,
        sector_scales: np.ndarray,
        wedge_size: int,
    ) -> None:
        full_to_wedge = np.asarray(full_to_wedge, dtype=np.int64)
        phases = np.asarray(phases, dtype=np.float64)
        sector_orbits = np.asarray(sector_orbits, dtype=np.int64)
        scales = np.asarray(sector_scales, dtype=np.float64)
        self.full_size = int(full_to_wedge.size)
        self.rows = int(sector_orbits.size)
        wedge_to_row = np.full(int(wedge_size), -1, dtype=np.int64)
        wedge_to_row[sector_orbits] = np.arange(self.rows, dtype=np.int64)
        point_rows = wedge_to_row[full_to_wedge]
        admitted = point_rows >= 0
        weights = np.zeros(self.full_size, dtype=np.float64)
        weights[admitted] = phases[admitted] * scales[point_rows[admitted]]
        self.point_rows = np.where(admitted, point_rows, 0)
        self.point_weights = weights
        # Points of every row, in increasing point order.
        points = np.flatnonzero(admitted)
        order = np.argsort(point_rows[points], kind="stable")
        points = points[order]
        rows_of = point_rows[points]
        counts = np.bincount(rows_of, minlength=self.rows)
        width = int(counts.max(initial=1))
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        slot = np.arange(points.size) - np.repeat(starts, counts)
        self.orbit_points = np.zeros((self.rows, width), dtype=np.int64)
        self.orbit_weights = np.zeros((self.rows, width), dtype=np.float64)
        self.orbit_points[rows_of, slot] = points
        self.orbit_weights[rows_of, slot] = weights[points]
        self._device: dict[int, tuple] = {}
        self._lock = Lock()

    def _arrays(self, cp):
        device = int(cp.cuda.Device().id)
        with self._lock:
            found = self._device.get(device)
            if found is None:
                found = tuple(
                    cp.asarray(array)
                    for array in (
                        self.point_rows,
                        self.point_weights,
                        self.orbit_points,
                        self.orbit_weights,
                    )
                )
                self._device[device] = found
        return found

    def unfold(self, cp, block):
        """``U_G C`` of a sector block on the current device (full grid x columns, F order)."""

        point_rows, point_weights, _, _ = self._arrays(cp)
        full = cp.asfortranarray(block[point_rows, :])
        full *= point_weights[:, None]
        return full

    def fold(self, cp, full):
        """``U_G^T F`` of a full-grid block on the current device (sector rows x columns)."""

        _, _, orbit_points, orbit_weights = self._arrays(cp)
        result = cp.zeros((self.rows, int(full.shape[1])), dtype=cp.float64, order="F")
        for slot in range(int(orbit_points.shape[1])):
            result += full[orbit_points[:, slot], :] * orbit_weights[:, slot, None]
        return result

    def device_bytes(self) -> int:
        return int(
            self.point_rows.nbytes
            + self.point_weights.nbytes
            + self.orbit_points.nbytes
            + self.orbit_weights.nbytes
        )


@dataclass
class OrbitalSector:
    """The orbitals of one channel that one sector (or the full grid) holds.

    ``vectors`` has the sector's rows (the full grid without symmetry) and at
    least the columns ``local``; ``states[k]`` is the channel state of local
    column ``local[k]``.  It lies on ``device`` or, spilled, on the host.
    ``expansion`` is ``None`` on the full grid.
    """

    representation: int | None
    device: int
    vectors: Any
    local: np.ndarray
    states: np.ndarray
    expansion: SectorExpansion | None

    @property
    def rows(self) -> int:
        return int(self.vectors.shape[0])

    def columns(self, cp, positions: np.ndarray):
        """The columns ``local[positions]`` on the current device and stream, F order.

        The basis is not written while the exchange is formed.  On the same
        device the columns are taken on the current stream.  From a basis on
        another device, an F-contiguous range goes as one peer copy on the
        current stream; any other selection -- and the eigensolvers hand
        C-ordered sector bases and strided full-grid blocks -- is first
        gathered into a temporary on the source device's gather stream.  The
        temporary's block returns to that stream's pool arena when it is
        dropped, so the gather stream waits for an event recorded after the
        copy on the current stream: no later gather can reuse the block while
        the copy still reads it.
        """

        selected = self.local[np.asarray(positions, dtype=np.int64)]
        vectors = self.vectors
        if vectors is None:
            raise RuntimeError("these orbitals were dropped after their exchange operator was built")
        current = int(cp.cuda.Device().id)
        if not isinstance(vectors, cp.ndarray):
            return cp.asarray(np.asfortranarray(np.asarray(vectors)[:, selected]), dtype=cp.float64)
        source_device = int(vectors.device.id)
        contiguous = self._contiguous(selected)
        view = vectors[:, int(selected[0]) : int(selected[-1]) + 1] if contiguous else None
        if source_device == current:
            if view is not None:
                return cp.asfortranarray(view)
            return cp.asfortranarray(vectors[:, cp.asarray(selected)])
        gather = None
        if view is None or not view.flags.f_contiguous:
            with cp.cuda.Device(source_device):
                with _GATHER_LOCK:
                    gather = _gather_stream(cp, source_device)
                    with gather:
                        view = cp.asfortranarray(
                            view if view is not None else vectors[:, cp.asarray(selected)]
                        )
                        gather.synchronize()
        copy = cp.empty(view.shape, dtype=cp.float64, order="F")
        copy.data.copy_from_device_async(view.data, view.nbytes)
        if gather is not None:
            # While ``view`` is alive: whatever the gather stream runs next
            # (and reuses the block of ``view`` for) comes after the copy.
            gather.wait_event(cp.cuda.get_current_stream().record())
        return copy

    @staticmethod
    def _contiguous(selected: np.ndarray) -> bool:
        return bool(selected.size) and bool(np.all(np.diff(selected) == 1))

    def products(self, cp, positions: np.ndarray, right, *, diagonal: bool = False):
        """``C^T R`` (or its diagonal, column by column) on the current device.

        ``C`` are the columns ``local[positions]`` and ``R`` a device block
        with the sector's rows.  A basis that was spilled to the host is
        uploaded in row chunks of at most 256 MiB, so that no complete copy
        of it is made on the device.
        """

        vectors = self.vectors
        if vectors is None:
            raise RuntimeError("these orbitals were dropped after their exchange operator was built")
        if isinstance(vectors, cp.ndarray):
            block = self.columns(cp, positions)
            return cp.sum(block * right, axis=0) if diagonal else block.T @ right
        selected = self.local[np.asarray(positions, dtype=np.int64)]
        host = np.asarray(vectors)
        rows = int(host.shape[0])
        step = max(1, _PRODUCT_CHUNK_BYTES // (8 * max(1, selected.size)))
        total = None
        for start in range(0, rows, step):
            stop = min(rows, start + step)
            block = cp.asarray(np.asfortranarray(host[start:stop][:, selected]), dtype=cp.float64)
            part = (
                cp.sum(block * right[start:stop], axis=0)
                if diagonal
                else block.T @ right[start:stop]
            )
            total = part if total is None else total + part
        return total

    def full_columns(self, cp, positions: np.ndarray):
        """``U_G`` of :meth:`columns` on the current device (full grid)."""

        block = self.columns(cp, positions)
        if self.expansion is None:
            return block
        return self.expansion.unfold(cp, block)


@dataclass
class ChannelOrbitals:
    """The orbitals of one spin channel (``spin_weight`` 1) or of both (2)."""

    sectors: tuple[OrbitalSector, ...]
    occupations: np.ndarray
    spin_weight: float
    full_size: int
    volume_element: float

    @property
    def state_count(self) -> int:
        return int(self.occupations.size)

    def occupied_states(self) -> np.ndarray:
        return np.flatnonzero(self.occupations > OCCUPATION_THRESHOLD)


def channel_orbitals(
    wavefunctions,
    occupations,
    spin_weight: float,
    volume_element: float,
    expansions=None,
    sector_devices=None,
):
    """:class:`ChannelOrbitals` of an eigensolver's orbitals.

    ``wavefunctions`` is a device (or host) full-grid block or the
    :class:`~parsec_python.acceleration.Eigensolvers.symmetry.CuPySymmetryOrbitals`
    of the sectors; ``expansions`` caches the :class:`SectorExpansion` of
    each representation (a dict filled here), ``sector_devices`` names the
    device of a spilled sector.
    """

    from ..backends.cupy import require_cupy
    from ..Eigensolvers.symmetry import CuPySymmetryOrbitals

    cp, _ = require_cupy()
    occupations = np.asarray(occupations, dtype=np.float64)
    expansions = {} if expansions is None else expansions
    if isinstance(wavefunctions, CuPySymmetryOrbitals):
        if wavefunctions.remote_sectors:
            raise NotImplementedError(
                "the exchange operator of a hybrid functional needs the orbitals of every "
                "symmetry sector in this process (no MPI ranks)"
            )
        for record in wavefunctions.sector_handovers or ():
            if record is not None and not getattr(record, "rotated", True):
                raise NotImplementedError(
                    "the exchange operator of a hybrid functional needs Ritz vectors; the "
                    "triangular Ritz hand-over (PARSEC_CUPY_RITZ_HANDOVER) is not supported with it"
                )
        if wavefunctions.scaled_wedge_vectors is not None or wavefunctions.sector_vectors is None:
            raise RuntimeError(
                "the symmetry orbitals of this iteration were released before the exchange "
                "operator read them"
            )
        sectors = []
        for representation, vectors in enumerate(wavefunctions.sector_vectors):
            selected = np.flatnonzero(wavefunctions.representations == representation)
            if not selected.size:
                continue
            if vectors is None or hasattr(vectors, "group"):
                raise NotImplementedError(
                    "the exchange operator of a hybrid functional needs every sector basis on "
                    "one device (a basis shared by several devices is not supported)"
                )
            expansion = expansions.get(representation)
            if expansion is None:
                expansion = SectorExpansion(
                    wavefunctions.full_to_wedge,
                    cp.asnumpy(wavefunctions.phases[representation])
                    if isinstance(wavefunctions.phases, cp.ndarray)
                    else np.asarray(wavefunctions.phases[representation]),
                    wavefunctions.sector_orbits[representation],
                    wavefunctions.sector_scales[representation],
                    int(wavefunctions.wedge_size),
                )
                expansions[representation] = expansion
            if isinstance(vectors, cp.ndarray):
                device = int(vectors.device.id)
            elif sector_devices is not None:
                device = int(sector_devices[representation])
            else:
                device = int(cp.cuda.Device().id)
            sectors.append(
                OrbitalSector(
                    representation=representation,
                    device=device,
                    vectors=vectors,
                    local=np.asarray(wavefunctions.representation_columns[selected], dtype=np.int64),
                    states=selected.astype(np.int64),
                    expansion=expansion,
                )
            )
        return ChannelOrbitals(
            tuple(sectors),
            occupations,
            float(spin_weight),
            int(wavefunctions.full_size),
            float(volume_element),
        )
    if getattr(wavefunctions, "ndim", 2) != 2:
        raise ValueError("orbitals must be a (grid, states) block")
    count = int(wavefunctions.shape[1])
    if count != occupations.size:
        raise ValueError("one occupation per orbital column is required")
    device = (
        int(wavefunctions.device.id)
        if isinstance(wavefunctions, cp.ndarray)
        else int(cp.cuda.Device().id)
    )
    sector = OrbitalSector(
        representation=None,
        device=device,
        vectors=wavefunctions,
        local=np.arange(count, dtype=np.int64),
        states=np.arange(count, dtype=np.int64),
        expansion=None,
    )
    return ChannelOrbitals(
        (sector,), occupations, float(spin_weight), int(wavefunctions.shape[0]), float(volume_element)
    )


# ---------------------------------------------------------------------------
# Pair solvers on every device.
# ---------------------------------------------------------------------------


class DevicePairSolvers:
    """The Kronecker pair solver of a hybrid's exchange kernel, uploaded per device.

    ``terms`` are ``(kernel, weight, omega)``.  One term solves its kernel
    with weight one (the fraction ``a_x`` scales the operator).  Two terms
    (CAM-B3LYP: ``0.19/r + 0.46 erf(w r)/r``) give the operator of fraction
    one with ``sum_t a_t v_t``, as
    :func:`parsec_python.SCF.hybrid.default_pair_solver` builds it: by
    default one solve of the weighted kernel ``a/r + b K(r)``
    (:class:`parsec_python.Hartree.kronecker.KroneckerPoisson` with
    ``weights``), with ``single_solve=False`` one solve per kernel, summed
    (the form of :class:`parsec_python.SCF.hybrid.WeightedPairPoisson`).
    The host plans are built once; ``batch_size`` densities make one solve.
    """

    def __init__(
        self,
        grid,
        terms,
        multipole_order: int,
        batch_size: int,
        *,
        single_solve: bool = True,
        background: bool = False,
    ) -> None:
        self.batch_size = max(1, int(batch_size))
        self.terms = tuple((str(kernel), float(weight), omega) for kernel, weight, omega in terms)
        self.multipole_order = int(multipole_order)
        self.setup_seconds = 0.0
        self._devices: dict[int, tuple] = {}
        self._streams: dict = {}
        self._lock = Lock()
        self._built = None
        # With ``background`` the host plans are built on a thread of their
        # own (NumPy only, nothing on a device), beside the semilocal start
        # of the Fock loop; the first use waits for them.
        self._future = None
        if background:
            from concurrent.futures import ThreadPoolExecutor as _Pool

            pool = _Pool(max_workers=1, thread_name_prefix="parsec-exx-plan")
            self._future = pool.submit(self._build, grid, single_solve)
            pool.shutdown(wait=False)
        else:
            self._built = self._build(grid, single_solve)

    def _build(self, grid, single_solve: bool):
        from parsec_python.Hartree.kronecker import KroneckerPoisson

        started = perf_counter()
        kernels = [kernel for kernel, _weight, _omega in self.terms]
        screened_kernels = {kernel for kernel in kernels if kernel != "coulomb"}
        if single_solve and len(self.terms) == 2 and "coulomb" in kernels and len(screened_kernels) == 1:
            coulomb = next(term for term in self.terms if term[0] == "coulomb")
            screened = next(term for term in self.terms if term[0] != "coulomb")
            host = KroneckerPoisson(
                grid,
                kernel=screened[0],
                omega=float(screened[2]),
                multipole_order=self.multipole_order,
                weights=(coulomb[1], screened[1]),
            )
            parts = ((1.0, host),)
            labels = [
                f"{coulomb[1]:g}/r + {screened[1]:g} {screened[0]}(w r)/r in one solve, "
                f"w = {float(screened[2]):g} bohr^-1"
            ]
        else:
            weighted = len(self.terms) > 1
            parts = tuple(
                (
                    weight if weighted else 1.0,
                    KroneckerPoisson(
                        grid,
                        kernel=kernel,
                        omega=None if kernel == "coulomb" else float(omega),
                        multipole_order=self.multipole_order,
                    ),
                )
                for kernel, weight, omega in self.terms
            )
            labels = [f"{weight:g} x {kernel}" for kernel, weight, _omega in self.terms] if weighted else []
        self.setup_seconds = perf_counter() - started
        base = str(getattr(parts[0][1], "description", "Kronecker-product FD solver")).replace(
            "Kronecker-product", "FP64 CUDA Kronecker-product", 1
        )
        description = base + (f"; {', '.join(labels)}" if labels else "") + (
            f"; devices take tiles of {self.batch_size}-column orbital blocks"
        )
        return parts, description

    def _ready(self):
        if self._built is None:
            self._built = self._future.result()
        return self._built

    @property
    def parts(self):
        return self._ready()[0]

    @property
    def hosts(self):
        return tuple(host for _weight, host in self.parts)

    @property
    def size(self) -> int:
        return int(self.hosts[0].size)

    @property
    def description(self) -> str:
        return self._ready()[1]

    def solvers(self, device: int):
        from ..Hartree.kronecker_device import WORK_ARRAYS_PER_DENSITY, DeviceKroneckerPoisson

        with self._lock:
            found = self._devices.get(int(device))
        if found is not None:
            return found
        per_density = 8 * (WORK_ARRAYS_PER_DENSITY * self.hosts[0].box_size + 3 * self.size)
        made = tuple(
            DeviceKroneckerPoisson(
                host, device_id=int(device), memory_budget=per_density * self.batch_size
            )
            for host in self.hosts
        )
        with self._lock:
            self._devices[int(device)] = made
        return made

    def stream(self, device: int, purpose: str = "work"):
        """A persistent non-blocking stream of ``device`` for the exchange (``work`` or ``delivery``).

        Kept for the run: CuPy's pool keeps the blocks freed on a stream for
        that stream, so the streams are reused and their blocks returned
        after every build (:meth:`release_blocks`).
        """

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        key = (int(device), purpose)
        with self._lock:
            stream = self._streams.get(key)
            if stream is None:
                with cp.cuda.Device(int(device)):
                    stream = cp.cuda.Stream(non_blocking=True)
                self._streams[key] = stream
        return stream

    def release_blocks(self) -> None:
        """Return the pool blocks the exchange streams freed, for the eigensolvers."""

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        with self._lock:
            streams = list(self._streams.items())
        for (device, _purpose), stream in streams:
            with cp.cuda.Device(device):
                stream.synchronize()
                cp.get_default_memory_pool().free_all_blocks(stream=stream)

    def plan_bytes(self) -> int:
        return int(sum(host.plan.nbytes() for host in self.hosts))

    def solve(self, cp, device: int, densities):
        """Potentials ``(full grid, nb)`` (Ry, F order) of densities on ``device``."""

        from parsec_python.Hartree.kronecker import solve_block

        total = None
        for (weight, _host), solver in zip(self.parts, self.solvers(device)):
            values = solve_block(solver.plan, densities, cp).T
            if weight != 1.0:
                values = weight * values
            total = values if total is None else total + values
        return total


# ---------------------------------------------------------------------------
# Device memory of the exchange.
# ---------------------------------------------------------------------------

#: Bytes per box point of a device Kronecker plan (eigenvector products,
#: spectral factors, face corrections; the split screening's exterior tails):
#: kron's measured 0.9 GB (Coulomb) and 3.0 GB (erfc) at n = 170, rounded up.
_PLAN_BYTES_PER_BOX_POINT = {"coulomb": 200, "screened": 650}
#: Work arrays per density of one solve (kronecker_device.WORK_ARRAYS_PER_DENSITY)
#: and full-grid arrays per tile column (two unfolded blocks, two accumulators,
#: densities, potentials and the solve's own three).
_WORK_PER_DENSITY = 8
_TILE_ARRAYS = 9


def estimated_box_points(grid, kernels, multipole_order: int) -> int:
    """Points of the Kronecker box of ``grid`` (bounding box and automatic padding, no plan built)."""

    import math

    from parsec_python.Hartree import kronecker

    try:
        integers = np.asarray(grid.integer_coordinates, dtype=np.int64)
        lower, upper = integers.min(axis=0), integers.max(axis=0)
        spacing = float(grid.spacing)
        source_squared = float(np.max(np.einsum("ij,ij->i", grid.coordinates, grid.coordinates)))
        smearing = 0.0
        if any(kernel != "coulomb" for kernel in kernels):
            argument = kronecker._smearing_exponent_argument(int(multipole_order))
            smearing = spacing * math.sqrt(argument / kronecker.DEFAULT_SMEARING_LIMIT)
        below, above = kronecker._automatic_padding(
            lower, upper, np.asarray(grid.settings.shift, dtype=np.float64), spacing, source_squared, smearing
        )
        shape = (upper + above) - (lower - below) + 1
        return int(np.prod(shape))
    except Exception:  # an estimate only: the bounding box of the grid
        integers = np.asarray(grid.integer_coordinates, dtype=np.int64)
        return int(np.prod(integers.max(axis=0) - integers.min(axis=0) + 1))


def planned_exchange_bytes(
    *,
    full_size: int,
    box_points: int,
    kernels,
    sector_rows: dict,
    sector_devices: dict,
    devices,
    states: int,
    channels: int,
    batch_size: int,
) -> dict[int, int]:
    """Device bytes the exchange of a hybrid keeps or takes beside the orbitals, by device.

    - ``xi`` of every channel (``W`` shares its buffer) on the device of each
      sector: rows times the sector's share of ``states`` (proportional to
      its rows, plus 8), 8 bytes each;
    - on every solving device: the sector expansions (32 bytes per full-grid
      point and sector), the Kronecker plan (:data:`_PLAN_BYTES_PER_BOX_POINT`
      times the box) and the work of one tile (``(9 N + 8 B) b`` doubles).

    ``sector_rows``/``sector_devices`` map each sector (``None``: the full
    grid) to its rows and device.  The grouped filter's work is not counted:
    it takes what the device has free (:func:`.chebyshev._exchange_block_filter`).
    """

    total_rows = max(1, sum(int(rows) for rows in sector_rows.values()))
    counts = {int(device): 0 for device in devices}
    for key, rows in sector_rows.items():
        device = int(sector_devices[key])
        share = int(math.ceil(int(states) * int(rows) / total_rows)) + 8
        counts[device] = counts.get(device, 0) + 8 * int(rows) * min(int(states), share) * int(channels)
    screened = any(kernel != "coulomb" for kernel in kernels)
    plan = _PLAN_BYTES_PER_BOX_POINT["screened" if screened else "coulomb"] * int(box_points)
    work = 8 * int(batch_size) * (_TILE_ARRAYS * int(full_size) + _WORK_PER_DENSITY * int(box_points))
    expansions = 32 * int(full_size) * sum(1 for key in sector_rows if key is not None)
    for device in devices:
        counts[int(device)] = counts.get(int(device), 0) + plan + work + expansions
    return counts


def require_exchange_memory(orbitals: ChannelOrbitals, solvers, devices) -> None:
    """Refuse, with the numbers, a build whose ``W`` cannot be held where its sectors are."""

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    needed: dict[int, int] = {}
    for sector in orbitals.sectors:
        needed[sector.device] = needed.get(sector.device, 0) + 8 * sector.rows * int(sector.local.size)
    work = 8 * int(solvers.batch_size) * _TILE_ARRAYS * int(orbitals.full_size)
    for device in devices:
        needed[int(device)] = needed.get(int(device), 0) + work
    for device, count in sorted(needed.items()):
        with cp.cuda.Device(device):
            free, total = cp.cuda.runtime.memGetInfo()
            available = int(free) + int(cp.get_default_memory_pool().free_bytes())
        if count > available:
            raise MemoryError(
                f"the exact exchange of this hybrid needs {count / 2**30:.1f} GiB on device {device} "
                f"(W = V_x Psi of its sectors and the tile work) and {available / 2**30:.1f} GiB of "
                f"{int(total) / 2**30:.1f} GiB are free there; run it on more GPUs (the symmetry "
                "sectors spread over them), with fewer States_Num, or a smaller Exx_Batch_Size"
            )


# ---------------------------------------------------------------------------
# W = V_x Psi.
# ---------------------------------------------------------------------------


_ROW_KERNEL_SOURCE = r"""
extern "C" __global__ void exchange_row(
    const double* __restrict__ second,      // q_j (n x nb, columns of stride n)
    const double* __restrict__ potentials,  // v_ij (n x nb)
    const double* __restrict__ first,       // q_i (n)
    const double* __restrict__ weights,     // f_j (nb)
    const double first_weight,              // f_i
    double* __restrict__ first_acc,         // W_i (n)
    double* __restrict__ second_acc,        // W_j (n x nb) or null
    const long long n,
    const int nb)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const double qi = first[p] * first_weight;
    double sum = 0.0;
    for (int j = 0; j < nb; ++j) {
        const double v = potentials[p + (long long)j * n];
        sum += weights[j] * second[p + (long long)j * n] * v;
        if (second_acc != nullptr) second_acc[p + (long long)j * n] -= qi * v;
    }
    first_acc[p] -= sum;
}
"""
_ROW_KERNELS: dict[int, Any] = {}


def _accumulate_row(cp, second, potentials, first, weights, first_weight, first_acc, second_acc) -> None:
    """``W_i -= sum_j f_j q_j v_ij`` and ``W_j -= f_i q_i v_ij`` in one pass over the grid.

    The arrays are F-contiguous blocks of the full grid on the current
    device (``first``/``first_acc`` contiguous columns); ``second_acc`` is
    ``None`` for an empty target row (no ``W_j`` update).  Every point sums
    its ``nb`` terms in a fixed order.
    """

    device = int(cp.cuda.Device().id)
    kernel = _ROW_KERNELS.get(device)
    if kernel is None:
        kernel = cp.RawKernel(_ROW_KERNEL_SOURCE, "exchange_row", options=("--fmad=false",))
        _ROW_KERNELS[device] = kernel
    count, width = int(second.shape[0]), int(second.shape[1])
    for array in (second, potentials) + (() if second_acc is None else (second_acc,)):
        if not array.flags.f_contiguous or array.shape != (count, width):
            raise ValueError("the row update needs F-contiguous (grid x batch) blocks")
    if not (first.flags.c_contiguous and first_acc.flags.c_contiguous):
        raise ValueError("the row update needs contiguous columns")
    threads = 256
    kernel(
        ((count + threads - 1) // threads,),
        (threads,),
        (
            second,
            potentials,
            first,
            cp.ascontiguousarray(weights, dtype=cp.float64),
            np.float64(first_weight),
            first_acc,
            second_acc if second_acc is not None else np.uintp(0),
            np.int64(count),
            np.int32(width),
        ),
    )


@dataclass
class DeviceExchangeApplication:
    """``W_G = U_G^T V_x Psi_G`` of every sector of one channel and what it took.

    ``applied[s]`` belongs to ``orbitals.sectors[s]`` (sector rows x its
    columns ``local``, on its device), with zero columns for the states not in
    ``columns``.  ``energy`` is ``E_x`` of the channel (Ry, unscaled).
    """

    orbitals: ChannelOrbitals
    applied: list
    columns: tuple[int, ...]
    occupied: tuple[int, ...]
    energy: float
    pair_solves: int
    poisson_seconds: float
    seconds: float


@dataclass
class _Block:
    sector: int
    positions: np.ndarray  # positions into the sector's local columns
    occupied: bool


@dataclass
class _Tile:
    first: int
    second: int
    pairs: int
    kind: str  # "diagonal", "pair" (two occupied blocks) or "empty" (empty first, occupied second)


def _blocks(orbitals: ChannelOrbitals, wanted: set[int], size: int):
    """Occupied blocks of every sector and the blocks of wanted empty states."""

    occupied_blocks: list[_Block] = []
    empty_blocks: list[_Block] = []
    for index, sector in enumerate(orbitals.sectors):
        weights = orbitals.occupations[sector.states]
        occupied = np.flatnonzero(weights > OCCUPATION_THRESHOLD)
        empty = np.array(
            [position for position in np.flatnonzero(weights <= OCCUPATION_THRESHOLD)
             if int(sector.states[position]) in wanted],
            dtype=np.int64,
        )
        for start in range(0, occupied.size, size):
            occupied_blocks.append(_Block(index, occupied[start : start + size], True))
        for start in range(0, empty.size, size):
            empty_blocks.append(_Block(index, empty[start : start + size], False))
    return occupied_blocks, empty_blocks


def apply_device_exchange(
    orbitals: ChannelOrbitals,
    solvers: DevicePairSolvers,
    devices: Sequence[int],
    *,
    columns: str = "all",
    known: DeviceExchangeApplication | None = None,
) -> DeviceExchangeApplication:
    """``W = V_x Psi`` of a channel on ``devices`` (see the module text).

    ``columns="occupied"`` forms the occupied columns only (enough for the
    energy); ``"all"`` every column.  ``known``, an application of the
    occupied columns of the same orbitals, is completed with the empty ones
    and consumed.
    """

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    started = perf_counter()
    occupied_states = [int(state) for state in orbitals.occupied_states()]
    all_states = set(range(orbitals.state_count))
    if columns not in ("all", "occupied"):
        raise ValueError("columns must be 'all' or 'occupied'")
    if known is not None:
        if known.orbitals is not orbitals or tuple(known.occupied) != tuple(occupied_states):
            raise ValueError("the known application belongs to other orbitals")
        if set(known.columns) != set(occupied_states):
            raise ValueError("a known application must hold the occupied columns")
    size = int(solvers.batch_size)
    wanted_empty = (all_states - set(occupied_states)) if columns == "all" else set()
    occupied_blocks, empty_blocks = _blocks(orbitals, wanted_empty, size)
    blocks = occupied_blocks + empty_blocks
    tiles: list[_Tile] = []
    if known is None:
        for first in range(len(occupied_blocks)):
            count = occupied_blocks[first].positions.size
            tiles.append(_Tile(first, first, count * (count + 1) // 2, "diagonal"))
            for second in range(first + 1, len(occupied_blocks)):
                tiles.append(
                    _Tile(first, second, count * occupied_blocks[second].positions.size, "pair")
                )
    for offset, empty in enumerate(empty_blocks):
        for second in range(len(occupied_blocks)):
            tiles.append(
                _Tile(
                    len(occupied_blocks) + offset,
                    second,
                    empty.positions.size * occupied_blocks[second].positions.size,
                    "empty",
                )
            )
    # W of every sector on its device.
    if known is not None:
        applied = known.applied
    else:
        require_exchange_memory(orbitals, solvers, devices)
        applied = []
        for sector in orbitals.sectors:
            with cp.cuda.Device(sector.device):
                applied.append(
                    cp.zeros((sector.rows, sector.local.size), dtype=cp.float64, order="F")
                )
    locks = [Lock() for _ in orbitals.sectors]
    # Most expensive tiles first, each to the least loaded device; a device
    # then runs its tiles grouped by their first block.
    devices = tuple(int(device) for device in devices) or (orbitals.sectors[0].device,)
    loads = {device: 0 for device in devices}
    assigned: dict[int, list[_Tile]] = {device: [] for device in devices}
    for tile in sorted(tiles, key=lambda item: -item.pairs):
        device = min(devices, key=lambda name: (loads[name], devices.index(name)))
        loads[device] += tile.pairs
        assigned[device].append(tile)
    for device in devices:
        assigned[device].sort(key=lambda item: (item.first, item.second))
    inverse_volume = 1.0 / float(orbitals.volume_element)
    timing = {device: 0.0 for device in devices}
    # The bases were written on the eigensolvers' streams and W was zeroed on
    # the current one: every stream of every device involved is done first.
    for device in sorted({sector.device for sector in orbitals.sectors} | set(devices)):
        cp.cuda.Device(device).synchronize()

    def deliver(block: _Block, part) -> None:
        """Add a folded part of ``W`` to its sector, on the sector's device.

        The part is ready when the worker's stream reaches the recorded
        event; the copy and the addition run on a stream of the sector's
        device that only deliveries use, one at a time (the lock), and the
        part is kept until they are done.
        """

        sector = orbitals.sectors[block.sector]
        target = applied[block.sector]
        positions = block.positions
        ready = cp.cuda.get_current_stream().record()
        with locks[block.sector]:
            with cp.cuda.Device(sector.device):
                stream = solvers.stream(sector.device, "delivery")
                with stream:
                    stream.wait_event(ready)
                    if int(part.device.id) == sector.device:
                        received = part
                    else:
                        received = cp.empty(part.shape, dtype=cp.float64, order="F")
                        received.data.copy_from_device_async(part.data, part.nbytes)
                    if OrbitalSector._contiguous(positions):
                        start, stop = int(positions[0]), int(positions[-1]) + 1
                        target[:, start:stop] += received
                    else:
                        index = cp.asarray(positions)
                        target[:, index] = target[:, index] + received
                stream.synchronize()

    def folded(block: _Block, full):
        sector = orbitals.sectors[block.sector]
        if sector.expansion is None:
            return cp.asfortranarray(full)
        return sector.expansion.fold(cp, full)

    def solve(device: int, densities, pending: list):
        """Potentials of ``densities``; the solve is timed by CUDA events."""

        begin = cp.cuda.Event()
        finish = cp.cuda.Event()
        begin.record()
        potentials = solvers.solve(cp, device, densities)
        finish.record()
        pending.append((begin, finish))
        return potentials

    def settle(device: int, pending: list) -> None:
        for begin, finish in pending:
            finish.synchronize()
            timing[device] += cp.cuda.get_elapsed_time(begin, finish) * 1.0e-3
        pending.clear()

    def run(device: int) -> None:
        work = assigned[device]
        if not work:
            return
        pending: list = []
        # Each device works on a stream of its own: what one device queues
        # never waits for what another queues.
        with cp.cuda.Device(device), solvers.stream(device):
            first_index = None
            first_full = first_acc = None
            first_weights = None
            for tile in work:
                if tile.first != first_index:
                    if first_acc is not None:
                        deliver(blocks[first_index], folded(blocks[first_index], first_acc))
                    first_index = tile.first
                    block = blocks[first_index]
                    sector = orbitals.sectors[block.sector]
                    first_full = sector.full_columns(cp, block.positions)
                    first_acc = cp.zeros(first_full.shape, dtype=cp.float64, order="F")
                    first_weights = orbitals.occupations[sector.states[block.positions]]
                second_block = blocks[tile.second]
                if tile.kind == "diagonal":
                    # Pairs i <= j of one block, Exx_Batch_Size at a time:
                    # W_i -= f_j q_j v_ij and, for i != j, W_j -= f_i q_i v_ij,
                    # added to the columns by one-hot products.
                    count = first_full.shape[1]
                    left, right = np.triu_indices(count)
                    weights = cp.asarray(first_weights)
                    left_device = cp.asarray(left)
                    right_device = cp.asarray(right)
                    for begin in range(0, left.size, size):
                        i = left_device[begin : begin + size]
                        j = right_device[begin : begin + size]
                        width = int(i.size)
                        first_columns = first_full[:, i]
                        second_columns = first_full[:, j]
                        densities = cp.asfortranarray(first_columns * second_columns)
                        densities *= inverse_volume
                        potentials = solve(device, densities, pending)
                        del densities
                        selector = cp.zeros((width, count), dtype=cp.float64)
                        rows = cp.arange(width)
                        gathered = second_columns * potentials
                        gathered *= weights[j][None, :]
                        selector[rows, i] = 1.0
                        first_acc -= gathered @ selector
                        mask = (i != j).astype(cp.float64)
                        gathered = first_columns * potentials
                        gathered *= (weights[i] * mask)[None, :]
                        selector.fill(0.0)
                        selector[rows, j] = 1.0
                        first_acc -= gathered @ selector
                        del gathered, potentials, first_columns, second_columns
                    settle(device, pending)
                    continue
                # A pair of blocks (or an empty block with an occupied one):
                # one solve per row i of the first block, its pairs with every
                # column j of the second, W_i -= sum_j f_j q_j v_ij (a GEMV)
                # and W_j -= f_i q_i v_ij.
                sector = orbitals.sectors[second_block.sector]
                second_full = sector.full_columns(cp, second_block.positions)
                second_weights = cp.asarray(
                    orbitals.occupations[sector.states[second_block.positions]]
                )
                second_acc = (
                    cp.zeros(second_full.shape, dtype=cp.float64, order="F")
                    if tile.kind == "pair"
                    else None
                )
                densities = cp.empty(second_full.shape, dtype=cp.float64, order="F")
                for row in range(first_full.shape[1]):
                    column = first_full[:, row : row + 1]
                    cp.multiply(second_full, column * inverse_volume, out=densities)
                    potentials = cp.asfortranarray(solve(device, densities, pending))
                    _accumulate_row(
                        cp,
                        second_full,
                        potentials,
                        first_full[:, row],
                        second_weights,
                        float(first_weights[row]),
                        first_acc[:, row],
                        second_acc,
                    )
                    del potentials
                settle(device, pending)
                del densities
                if tile.kind == "pair":
                    deliver(second_block, folded(second_block, second_acc))
                del second_full, second_acc
            if first_acc is not None:
                deliver(blocks[first_index], folded(blocks[first_index], first_acc))

    try:
        if len(devices) == 1:
            run(devices[0])
        else:
            with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="parsec-exx") as pool:
                for future in [pool.submit(run, device) for device in devices]:
                    future.result()
    finally:
        solvers.release_blocks()
        release_gather_blocks(cp)
    energy = 0.0
    for index, sector in enumerate(orbitals.sectors):
        weights = orbitals.occupations[sector.states]
        occupied = np.flatnonzero(weights > OCCUPATION_THRESHOLD)
        if not occupied.size:
            continue
        with cp.cuda.Device(sector.device):
            values = sector.products(cp, occupied, applied[index][:, occupied], diagonal=True)
            energy += float(cp.asnumpy(values) @ weights[occupied])
    energy *= 0.5 * orbitals.spin_weight
    pair_solves = sum(tile.pairs for tile in tiles)
    done = set(occupied_states) | (wanted_empty if columns == "all" else set())
    return DeviceExchangeApplication(
        orbitals=orbitals,
        applied=applied,
        columns=tuple(sorted(done)),
        occupied=tuple(occupied_states),
        energy=float(energy) if known is None else float(known.energy),
        pair_solves=int(pair_solves + (0 if known is None else known.pair_solves)),
        poisson_seconds=float(max(timing.values(), default=0.0) + (0.0 if known is None else known.poisson_seconds)),
        seconds=float(perf_counter() - started + (0.0 if known is None else known.seconds)),
    )


# ---------------------------------------------------------------------------
# ACE.
# ---------------------------------------------------------------------------


@dataclass
class DeviceCompressedExchange:
    """``a_x V_ACE = -a_x xi xi^T`` of one channel, ``xi_G`` per sector.

    ``projectors[representation]`` is ``xi_G`` on the sector's device
    (``None`` for the full grid key).  ``rank`` counts every column,
    ``energy`` is ``E_x`` of the orbitals it was built from (Ry, unscaled).
    """

    projectors: dict
    fraction: float
    energy: float
    spin_weight: float
    asymmetry: float
    pair_solves: int
    seconds: float
    rank: int = 0
    released: bool = False

    def release(self) -> None:
        """Free the device memory of ``xi``; the scalars stay."""

        self.projectors = {}
        self.released = True

    def band_energy(self, orbitals: ChannelOrbitals, *, scaled: bool = True) -> float:
        """``w sum_i f_i <q_i|a_x V_ACE|q_i>`` of ``orbitals`` (Ry).

        ``scaled=False`` without ``a_x``: ``2 E_x(P, P0)`` of the orbitals
        ``P`` with this operator of ``P0``.
        """

        from ..backends.cupy import require_cupy

        if self.released:
            raise RuntimeError("the compressed exchange operator was released")
        cp, _ = require_cupy()
        factor = -(self.fraction if scaled else 1.0)
        total = 0.0
        for sector in orbitals.sectors:
            xi = self.projectors.get(sector.representation)
            if xi is None:
                continue
            weights = orbitals.occupations[sector.states]
            occupied = np.flatnonzero(weights > OCCUPATION_THRESHOLD)
            if not occupied.size:
                continue
            with cp.cuda.Device(int(xi.device.id)):
                overlaps = sector.products(cp, occupied, xi)
                values = cp.sum(overlaps * overlaps, axis=1)
                total += float(cp.asnumpy(values) @ weights[occupied])
        return float(self.spin_weight * factor * total)


def _right_solve_lower_transpose(cp, lower, block) -> None:
    """``block <- block L^-T`` in place (``block`` F-contiguous, rows x k)."""

    from cupy.cuda import device as cuda_device
    from cupy_backends.cuda.libs import cublas

    lower = cp.asfortranarray(lower)
    rows, count = int(block.shape[0]), int(block.shape[1])
    if not block.flags.f_contiguous:
        raise ValueError("the block of the triangular solve must be F-contiguous")
    if rows == 0 or count == 0:
        return
    handle = cuda_device.get_cublas_handle()
    one = np.array(1.0, dtype=np.float64)
    mode = cublas.getPointerMode(handle)
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    try:
        cublas.dtrsm(
            handle,
            cublas.CUBLAS_SIDE_RIGHT,
            cublas.CUBLAS_FILL_MODE_LOWER,
            cublas.CUBLAS_OP_T,
            cublas.CUBLAS_DIAG_NON_UNIT,
            rows,
            count,
            one.ctypes.data,
            lower.data.ptr,
            count,
            block.data.ptr,
            rows,
        )
    finally:
        cublas.setPointerMode(handle, mode)


def compress_device_exchange(
    application: DeviceExchangeApplication, *, fraction: float = 1.0
) -> DeviceCompressedExchange:
    """``xi_G = W_G L_G^-T`` with ``M_G = -Psi_G^T W_G = L_G L_G^T`` per sector.

    ``W`` is overwritten by ``xi`` (the application is consumed).  ``M_G``
    is symmetrized; its asymmetry ``max|M - M^T| / max|M|`` over the sectors
    is reported.
    """

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    orbitals = application.orbitals
    if len(application.columns) != orbitals.state_count:
        raise ValueError("ACE needs V_x applied to every orbital of the block")
    started = perf_counter()
    projectors: dict = {}
    asymmetry = 0.0
    rank = 0
    for index, sector in enumerate(orbitals.sectors):
        applied = application.applied[index]
        with cp.cuda.Device(sector.device):
            overlap = -sector.products(cp, np.arange(sector.local.size), applied)
            scale = float(cp.abs(overlap).max()) if overlap.size else 0.0
            if scale > 0.0:
                asymmetry = max(asymmetry, float(cp.abs(overlap - overlap.T).max()) / scale)
            overlap = 0.5 * (overlap + overlap.T)
            try:
                lower = cp.linalg.cholesky(overlap)
            except Exception as error:  # cuSOLVER reports through LinAlgError or NaN
                raise np.linalg.LinAlgError(
                    "the exchange matrix M = -Psi^T V_x Psi of a symmetry sector is not "
                    "positive definite; ACE needs orbitals that overlap the occupied ones"
                ) from error
            if not bool(cp.all(cp.isfinite(lower))):
                raise np.linalg.LinAlgError(
                    "the exchange matrix M = -Psi^T V_x Psi of a symmetry sector is not "
                    "positive definite; ACE needs orbitals that overlap the occupied ones"
                )
            if not applied.flags.f_contiguous:
                applied = cp.asfortranarray(applied)
            _right_solve_lower_transpose(cp, lower, applied)
            cp.cuda.get_current_stream().synchronize()
        projectors[sector.representation] = applied
        application.applied[index] = None
        rank += int(sector.local.size)
    return DeviceCompressedExchange(
        projectors=projectors,
        fraction=float(fraction),
        energy=float(application.energy),
        spin_weight=float(orbitals.spin_weight),
        asymmetry=float(asymmetry),
        pair_solves=int(application.pair_solves),
        seconds=float(application.seconds + perf_counter() - started),
        rank=rank,
    )


__all__ = [
    "ChannelOrbitals",
    "DeviceCompressedExchange",
    "DeviceExchangeApplication",
    "DevicePairSolvers",
    "OrbitalSector",
    "SectorExpansion",
    "apply_device_exchange",
    "channel_orbitals",
    "compress_device_exchange",
    "estimated_box_points",
    "planned_exchange_bytes",
    "require_exchange_memory",
]
