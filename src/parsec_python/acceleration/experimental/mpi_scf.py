"""Root-controlled SCF with persistent, independently owned symmetry sectors.

Only rank zero enters the production SCF loop. Other ranks call worker_loop.
The existing first/subsequent eigensolver policies remain on each sector's
owner. The world communicator carries scalar potentials/densities and small
Ritz metadata, never orbital matrices. With one rank per node a sector stays on
the GPUs of its own rank; several of them share its basis by default
(Eigensolvers/distributed_state.py), otherwise the optional node-local filtering
gathers onto its owner GPU. The sectors go to the ranks and GPUs by their cost
(``PARSEC_SECTOR_PLACEMENT=size``, sector_rank_plan; ``cyclic`` deals them by
count as before), the root taking the lightest share. With one rank per GPU
(``gpu_per_rank``) and more ranks than sectors, disjoint groups of consecutive
ranks share the sectors, one or (after a CHEBFF first solve) several sectors per
group, solved one after another, which may span nodes: each group's communicator carries the
filter, layout changes, Gram sums and rotations of its basis
(Eigensolvers/mpi_shared_basis.py), and a run without symmetry is the case of one
sector. Only a CHEBFF first solve divides the columns of a sector among the ranks
of its group; after any other (CHEBDAV) every rank of the group holds and solves
the whole basis, and its density and force terms count on the group's first rank
alone (MPISectorContext.whole_basis_sectors). A hybrid functional keeps its own
ownership rules.
"""
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

import numpy as np


class MPISCFError(RuntimeError):
    """A worker failed; the outer launcher must abort the MPI job."""


def sector_rank_groups(representations, size):
    """The ranks of each sector: balanced groups of consecutive ranks where there are more ranks than sectors.

    With no more ranks than sectors, the sectors go round the ranks, one rank each (the former assignment).
    Otherwise every sector gets ``size // representations`` consecutive ranks and the first
    ``size % representations`` sectors one more, so that every rank serves one sector; consecutive ranks
    keep a group on as few nodes as the counts allow, and a group may span nodes.  This is the placement by
    count (``PARSEC_SECTOR_PLACEMENT=cyclic``); :func:`sector_rank_plan` places by size.
    """
    representations, size = int(representations), int(size)
    if representations < 1 or size < 1:
        raise ValueError('MPI ranks and representations must be positive')
    if size <= representations:
        return tuple((sector % size,) for sector in range(representations))
    base, extra = divmod(size, representations)
    edges = [0]
    for sector in range(representations):
        edges.append(edges[-1] + base + (sector < extra))
    return tuple(tuple(range(edges[sector], edges[sector + 1])) for sector in range(representations))


# Plans of groups of ranks whose fullest group holds within this share of the best one are taken as equal, and the
# one with the most groups (fewer ranks per basis, less exchange) is taken: the weighted sectors of a diamond cell
# differ by 1-6 % in rows, and five of them on sixteen ranks keep five groups (1/3 of a basis per GPU) instead of
# one group of all sixteen ranks (5/16).
_PLACEMENT_TIE = 0.10
# The device pool of a rank (every sector over all its GPUs, solved one after another) is taken where its fullest
# device holds less than 1 - this share of what disjoint device groups leave on theirs: three sectors on four GPUs
# (0.75 of a basis per GPU instead of 1, as before) or five on four (1.25 instead of 2), not sectors of nearly equal
# sizes that divide the GPUs.
_POOL_GAIN = 0.05


def sector_placement_setting():
    """``PARSEC_SECTOR_PLACEMENT``: ``size`` (the default) or ``cyclic``.

    ``size`` places the sectors by their cost (rows times first states, :func:`sector_rank_plan` and
    :func:`sector_device_groups`): the fullest rank and GPU hold as little as the sectors allow, the root (rank 0,
    which also holds the fields of the SCF loop) the lightest share, ranks of one GPU each may solve several sectors
    in a group one after another, and a rank may spread more sectors than it has GPUs over all of them.  ``cyclic``
    deals them by count as before 2026-11-19: sector ``s`` to rank ``s % size``, equal groups of ranks.  The
    sectors and their state counts are the same either way; only the ranks and GPUs that solve them differ, and so
    the order of the MPI sums.
    """
    import os

    value = os.environ.get('PARSEC_SECTOR_PLACEMENT', 'size').strip().lower() or 'size'
    if value not in {'size', 'cyclic'}:
        raise ValueError(f'PARSEC_SECTOR_PLACEMENT must be size or cyclic, not {value!r}')
    return value


def _placement_costs(representations, costs):
    """The cost of every sector as floats (all equal where ``costs`` is ``None``)."""
    if costs is None:
        return (1.0,) * int(representations)
    values = tuple(float(cost) for cost in costs)
    if len(values) != int(representations) or not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError(f'sector costs must be {int(representations)} positive numbers, not {costs!r}')
    return values


def _group_sizes(ranks, count):
    """``ranks`` ranks in ``count`` groups of consecutive ranks, the first ``ranks % count`` one larger."""
    return tuple(ranks // count + (1 if group < ranks % count else 0) for group in range(count))


def _fill_groups(costs, sizes):
    """The sectors of every group (longest first) and their summed costs, ``None`` where a group gets none.

    Each sector, the costliest first, goes to the group whose ranks then hold the least of the sum per rank;
    on a tie to a group without a sector, then to the first.  With equal costs and groups of one rank each,
    sector ``s`` goes to group ``s % len(sizes)``, the placement by count.
    """
    order = sorted(range(len(costs)), key=lambda sector: (-costs[sector], sector))
    members = [[] for _ in sizes]
    loads = [0.0] * len(sizes)
    for sector in order:
        group = min(range(len(sizes)),
                    key=lambda index: ((loads[index] + costs[sector]) / sizes[index], bool(members[index]), index))
        members[group].append(sector)
        loads[group] += costs[sector]
    if any(not items for items in members):
        return None
    return [sorted(items) for items in members], loads


def plan_sector_groups(costs, ranks, counts):
    """Groups of consecutive ranks and the sectors each solves, for the group counts in ``counts``.

    For every count the ranks are split into groups of near-equal sizes (:func:`_group_sizes`) and the sectors
    are filled in by cost (:func:`_fill_groups`); the plan whose ranks hold the least cost at the fullest group
    (sum of its sectors over its ranks) is taken, and of those within :data:`_PLACEMENT_TIE` of it the one with
    the most groups (fewer ranks per basis, less exchange; the k-point pools choose alike,
    ``SCF/kpoint_pools.plan_pools_by_size``).  The first ranks, which hold the root, take the group that holds the
    least per rank (the first on a tie, so that equal groups keep their order); the others follow in their order.
    Returns the ranks of every sector (a tuple of tuples).
    """
    costs = tuple(float(cost) for cost in costs)
    ranks = int(ranks)
    plans = []
    for count in counts:
        count = int(count)
        if not 1 <= count <= min(len(costs), ranks):
            continue
        sizes = _group_sizes(ranks, count)
        filled = _fill_groups(costs, sizes)
        if filled is None:
            continue
        members, loads = filled
        plans.append((max(load / width for load, width in zip(loads, sizes)), count, sizes, members, loads))
    if not plans:
        raise ValueError(f'no placement of {len(costs)} sectors on {ranks} ranks in {tuple(counts)} groups')
    best = min(plan[0] for plan in plans)
    _, count, sizes, members, loads = max(
        (plan for plan in plans if plan[0] <= best * (1.0 + _PLACEMENT_TIE) + 1.0e-300), key=lambda plan: plan[1])
    lightest = min(range(count), key=lambda group: (loads[group] / sizes[group], group))
    order = [lightest] + [group for group in range(count) if group != lightest]
    result = [None] * len(costs)
    start = 0
    for group in order:
        width = sizes[group]
        for sector in members[group]:
            result[sector] = tuple(range(start, start + width))
        start += width
    return tuple(result)


def sector_rank_plan(representations, size, *, shared=False, exchange=False, costs=None, divided=False):
    """The ranks that solve every sector (a tuple of rank tuples, one per sector), as ``configure`` places them.

    ``exchange`` (a hybrid functional) keeps sector ``s`` on rank ``s % size``.  With ``PARSEC_SECTOR_PLACEMENT``
    ``cyclic`` the placement is that by count: rank ``s % size``, or the equal groups of :func:`sector_rank_groups`
    where ranks share the sectors (``shared``).  With ``size`` (the default) the sectors are placed by ``costs``
    (:func:`plan_sector_groups`): without ``shared`` every rank gets one group of sectors (as many groups as
    ranks, the root the lightest); with ``shared`` groups of ranks share them, and where the first solve divides
    each basis among the ranks of its group (``divided``: CHEBFF, ``mpi_shared_basis.SharedBasis``) a group may
    solve several sectors one after another, which spreads ten sectors over sixteen ranks as two groups of eight
    with five each (5/8 of a basis per GPU) instead of six pairs and four single ranks (a whole basis on one GPU).
    After another first solve every rank of a group holds the whole bases of its group, so groups keep one sector
    each in the shape by count.
    """
    representations, size = int(representations), int(size)
    if representations < 1 or size < 1:
        raise ValueError('MPI ranks and representations must be positive')
    if exchange:
        return tuple((sector % size,) for sector in range(representations))
    if representations == 1:
        return (tuple(range(size)),)
    if sector_placement_setting() == 'cyclic':
        return sector_rank_groups(representations, size)
    costs = _placement_costs(representations, costs)
    if not shared:
        if size > representations:
            raise ValueError('MPI ranks must be between one and the representation count')
        return plan_sector_groups(costs, size, (size,))
    if not divided:
        if size < representations:
            raise ValueError('a shared sector needs at least one rank per sector')
        return sector_rank_groups(representations, size)
    return plan_sector_groups(costs, size, range(1, min(representations, size) + 1))


def _rank_device_groups(owned, costs, devices, pool=True):
    """The devices of each sector of one rank, by cost (``PARSEC_SECTOR_PLACEMENT=size``).

    At most as many sectors as devices take disjoint groups of devices in proportion to their costs (each a
    device more where its share per device is the largest; equal costs give the equal groups of before), more
    sectors than devices go to the devices longest first (equal costs: round the devices, as before).  Where
    spreading every sector over all the devices (the device pool, ``PARSEC_SECTOR_DEVICE_POOL``: solved one after
    another, each basis split over every device) leaves the fullest device holding less than
    ``1 - _POOL_GAIN`` of the disjoint groups' fullest, the sectors take the pool: three sectors on four
    devices (as before), and now also five on four (1.25 of a basis per device instead of 2).

    ``pool`` says whether the first solve divides the basis of every sector over the devices of a group
    (:func:`sector_device_groups`).  A basis that is not divided lies whole on the first device of its group, so
    the pool would put every sector there, one after another, and leave the other devices idle: without ``pool``
    the sectors keep the disjoint groups.
    """
    owned = tuple(owned)
    if not owned:
        return {}
    if len(devices) == 1:
        return {sector: (devices[0],) for sector in owned}
    if len(owned) <= len(devices):
        widths = [1] * len(owned)
        for _ in range(len(devices) - len(owned)):
            widest = max(range(len(owned)), key=lambda index: (costs[owned[index]] / widths[index], -index))
            widths[widest] += 1
        groups, start = [], 0
        for width in widths:
            groups.append(tuple(devices[start:start + width]))
            start += width
        fullest = max(costs[sector] / width for sector, width in zip(owned, widths))
        disjoint = dict(zip(owned, groups))
    else:
        members, loads = _fill_groups(tuple(costs[sector] for sector in owned), (1,) * len(devices))
        disjoint = {owned[index]: (devices[device],) for device, items in enumerate(members) for index in items}
        fullest = max(loads)
    pooled = sum(costs[sector] for sector in owned) / len(devices)
    if (pool and len(owned) > 1 and sector_device_pool_setting() == 'auto'
            and pooled < (1.0 - _POOL_GAIN) * fullest):
        return {sector: tuple(devices) for sector in owned}
    return {sector: disjoint[sector] for sector in owned}


def sector_device_groups(representations, size, rank, devices, shared=False, exchange=False, costs=None,
                         divided=False, divisible=None):
    """The devices of every sector that rank ``rank`` solves (``{sector: devices}``).

    ``PARSEC_SECTOR_PLACEMENT=cyclic`` assigns sectors cyclically to ranks and disjoint GPU groups when possible,
    as before 2026-11-19; ``size`` (the default) places them by ``costs`` (:func:`sector_rank_plan` for the
    ranks, :func:`_rank_device_groups` for the devices of a rank).  With ``shared`` every sector belongs to a group
    of ranks (the one sector of a run without symmetry to all of them), each of which holds a share of its basis
    on its first device (:mod:`..Eigensolvers.mpi_shared_basis`); ``divided`` says that the first solve divides
    the bases (CHEBFF), so that a group may solve several sectors.  Placed by size, the GPUs of a rank take the
    device pool only where its bases are divided over them: ``divided``, and every sector of the rank marked in
    ``divisible`` (one flag per sector, ``None`` for all: whether a CHEBFF first solve shares the basis over the
    devices of a group, ``Eigensolvers/symmetry.sector_bases_divisible``, which leaves below the threshold of the
    generalized Ritz step the basis on one device).  With ``exchange`` (a hybrid
    functional, ``SCF/hybrid_mpi.py``) every sector basis lies on one device
    of its rank, which the exchange operator needs, and a rank may own no
    sector: every device of every rank solves pair densities of the exchange.
    """
    representations, size, rank = int(representations), int(size), int(rank)
    devices = tuple(int(device) for device in devices)
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        raise ValueError('devices must be a nonempty list of distinct nonnegative indices')
    if exchange:
        if representations < 1 or size < 1 or not 0 <= rank < size:
            raise ValueError('MPI ranks and representations must be positive')
        owned = tuple(range(rank, representations, size))
        return {sector: (devices[index % len(devices)],) for index, sector in enumerate(owned)}
    cyclic = sector_placement_setting() == 'cyclic'
    if shared:
        if representations < 1 or not 0 <= rank < size or (size < representations and (cyclic or not divided)):
            raise ValueError('a shared sector needs at least one rank per sector')
        plan = sector_rank_plan(representations, size, shared=True, costs=costs, divided=divided)
        return {sector: (devices[0],) for sector, ranks in enumerate(plan) if rank in ranks}
    if representations < 1 or not 1 <= size <= representations or not 0 <= rank < size:
        raise ValueError('MPI ranks must be between one and the representation count')
    if not cyclic:
        plan = sector_rank_plan(representations, size, costs=costs)
        owned = tuple(sector for sector, ranks in enumerate(plan) if rank in ranks)
        if divisible is not None and len(divisible) != representations:
            raise ValueError(f'divisible must name {representations} sectors, not {divisible!r}')
        # After a CHEBDAV first solve, or for sectors below the generalized-Ritz threshold, every basis lies whole
        # on the first device of its group: the pool would solve all of them there (review of 2026-11-19).
        pool = bool(divided) and (divisible is None or all(bool(divisible[sector]) for sector in owned))
        return _rank_device_groups(owned, _placement_costs(representations, costs), devices, pool=pool)
    owned = tuple(range(rank, representations, size))
    if len(devices) < len(owned):
        return {sector: (devices[index % len(devices)],)
                for index, sector in enumerate(owned)}
    groups = np.array_split(devices, len(owned))
    if (len(owned) > 1 and len(devices) % len(owned) and sector_device_pool_setting() == 'auto'
            and len(owned) / len(devices) < 1.0 / min(len(group) for group in groups)):
        # Sectors that do not divide the devices evenly (3 on 4: groups of 2, 1 and 1 device) leave a device with
        # a whole sector; every sector over all the devices instead (solved one after another, their bases split
        # over every device) holds len(owned)/len(devices) of a sector per device (PARSEC_SECTOR_DEVICE_POOL).
        return {sector: tuple(int(x) for x in devices) for sector in owned}
    return {sector: tuple(int(x) for x in group)
            for sector, group in zip(owned, groups, strict=True)}


def force_term_signature(terms, labels):
    """``(shape, digest)`` of the nonlocal force terms of a rank: what the ranks compare before they sum them.

    ``labels`` are the projector labels ``(atom, l, m)`` that the rows of ``terms`` belong to (the
    ``projector_force_labels`` of the sector eigensolver: one atom per orbit of the group of the sectors where the
    projectors are compressed, ``Symmetry.axis_reflection.compress_gradients_for_sectors``); ``None`` where the
    solver names none.  The digest is a BLAKE2b of their integers, so that two ranks agree only where every row is
    the same projector.
    """
    import hashlib

    shape = tuple(int(size) for size in np.shape(terms))
    if labels is None:
        return shape, None
    labels = tuple(labels)
    try:
        data = np.asarray(labels, dtype=np.int64)
        content = repr(data.shape).encode() + data.tobytes()
    except (TypeError, ValueError):
        content = repr(tuple(tuple(int(value) for value in label) for label in labels)).encode()
    return shape, hashlib.blake2b(content, digest_size=16).hexdigest()


def force_signature_mismatches(signatures):
    """One entry per rank: ``None`` where its :func:`force_term_signature` is that of rank 0, else what differs."""
    first = signatures[0]
    return [None if item == first else
            f'its nonlocal force terms (rows {item[0]}, projector labels {item[1]}) are not those of rank 0 '
            f'(rows {first[0]}, projector labels {first[1]})' for item in signatures]


def sector_device_pool_setting():
    """``PARSEC_SECTOR_DEVICE_POOL``: ``auto`` (default) spreads every sector of a rank over all its devices where
    the sectors do not divide them evenly (:func:`sector_device_groups`); ``off`` keeps disjoint groups, as before."""
    import os

    value = os.environ.get('PARSEC_SECTOR_DEVICE_POOL', 'auto').strip().lower() or 'auto'
    if value not in {'auto', 'off'}:
        raise ValueError(f'PARSEC_SECTOR_DEVICE_POOL must be auto or off, not {value!r}')
    return value


class MPISectorContext:
    """MPI command protocol; all MPI calls occur on the calling main thread.

    Construct/configure on every rank. Preparation with this context builds
    the SCF-loop components (ionic fields, densities, Hartree, XC) on root
    only; non-root ranks enter worker_loop(solver, unwrapped_density_builder).
    Root wraps that builder with MPISymmetryDensityBuilder and runs the normal
    SCF. Call stop_workers after success. Any exception requires MPI.Abort;
    stop_workers deliberately does not communicate after a collective failure.
    One failure is not collective: a pool release that a rank waits for behind
    the collective calls of its density command (_release_joined) is raised by
    that rank alone, and the others learn of it by the launcher's Abort.

    ``exchange`` (a hybrid functional): every sector basis on one device, a
    rank may own no sector, every rank keeps the sector orbitals of its last
    solve (``_mpi_retained_results`` of the solver) for the exchange, and the
    commands ``exx_*`` go to the rank's ``exchange_service``
    (``SCF/hybrid_mpi.py``).  Every command carries the spin channel it is
    for (the active one of the root's solver, or the channel of the orbitals
    of a density); a rank serves both channels of a spin-polarized loop in
    the sectors it owns.

    The commands ``forces`` and ``spin_forces`` sum one row per projector of
    every rank on the root; the rows of a cell in the sectors of a point group
    belong to the projectors of one atom per orbit (``projector_force_labels``).
    Before the sum the ranks compare the shape of their rows and a digest of
    those labels (:func:`force_term_signature`): a difference fails the
    command on every rank instead of summing other projectors or leaving the
    sum waiting.  After the sum ``command_observer`` is called on every rank.
    """
    def __init__(self, comm, *, exchange=False):
        self.comm = comm
        self.rank, self.size = int(comm.rank), int(comm.size)
        self.root = 0
        self.failed = False
        self.stopped = False
        self.owned_sectors = ()
        self.device_groups = {}
        # Whether groups of ranks share the sector bases (configure), and the communicator of this rank's group.
        self.shared = False
        self.sector_comm = comm
        self.rank_groups = ()
        self.sector_owners = ()
        # One GPU per rank (the runner's --rank-per-gpu launch): more ranks than several sectors then share them.
        self.gpu_per_rank = False
        # Whether the first solve divides each shared basis among the ranks of its group (a CHEBFF first solve,
        # mpi_shared_basis.SharedBasis; the runner sets it from the input): only then may a group of ranks solve
        # several sectors (sector_rank_plan).
        self.divided_bases = False
        # The Ritz step of a shared basis where PARSEC_MPI_SHARED_RITZ names none
        # (mpi_shared_basis.shared_ritz_route): the runner asks for rounds for a periodic cell.
        self.shared_ritz_route = "rows"
        self.command_counts = {}
        # Wall seconds of each command on this rank, collective calls included.
        self.command_seconds = {}
        self.exchange = bool(exchange)
        self.retain_orbitals = bool(exchange)
        self.exchange_service = None
        # Called on every rank after each forces or spin_forces command with the command and how many of it this
        # rank has served (the full-SCF runner records the memory of its devices there); None calls nothing.
        self.command_observer = None

    # configure, planned_groups and divides take the costs of the sectors (Eigensolvers/symmetry.py,
    # sector_placement_costs; the sector eigensolver passes them where this is True), and configure and
    # planned_groups whether a CHEBFF first solve divides each basis over the devices of a group
    # (``divisible``, symmetry.sector_bases_divisible; None for every sector).
    accepts_sector_costs = True

    def planned_groups(self, representations, devices, costs=None, divisible=None):
        """The device groups of the sectors this rank will own (what ``configure`` assigns)."""
        return sector_device_groups(representations, self.size, self.rank, devices,
                                    shared=self._shared(representations, costs), exchange=self.exchange,
                                    costs=costs, divided=self.divided_bases, divisible=divisible)

    def divides(self, representations, costs=None):
        """Whether :meth:`configure` can give ``representations`` sectors to the ranks (no MPI call).

        A rank with the GPUs of its node owns one sector or more, so more ranks than several sectors need one GPU
        per rank (``gpu_per_rank``: groups of ranks share a sector) or a hybrid's exchange (ranks without a
        sector).  The driver keeps the group of before where a larger one has too few sectors for the ranks.
        Placing the sectors by size (``PARSEC_SECTOR_PLACEMENT``) changes which ranks solve them, not whether
        they can: the same counts divide either way.
        """
        try:
            sector_device_groups(representations, self.size, self.rank, (0,),
                                 shared=self._shared(representations, costs), exchange=self.exchange,
                                 costs=costs, divided=self.divided_bases)
        except ValueError:
            return False
        return True

    def _shared(self, representations, costs=None):
        # More ranks than sectors share their bases: the one sector of a run without symmetry, or several with
        # one GPU per rank (not with a hybrid).  Placed by size, ranks of one GPU each whose first solve divides
        # the bases also share them where groups that solve several sectors balance better than one rank each
        # (ten sectors on four ranks: two groups of two with five sectors, 2.5 bases per GPU instead of 3).
        representations = int(representations)
        if self.exchange or self.size <= 1:
            return False
        if representations == 1 or (self.gpu_per_rank and self.size > representations):
            return True
        if not self.gpu_per_rank or not self.divided_bases or sector_placement_setting() == 'cyclic':
            return False
        plan = plan_sector_groups(_placement_costs(representations, costs), self.size, range(1, self.size + 1))
        return any(len(ranks) > 1 for ranks in plan)

    def configure(self, representations, devices, costs=None, divisible=None):
        error = None
        try:
            shared = self._shared(representations, costs)
            groups = sector_device_groups(representations, self.size, self.rank, devices, shared=shared,
                                          exchange=self.exchange, costs=costs, divided=self.divided_bases,
                                          divisible=divisible)
            rank_groups = sector_rank_plan(representations, self.size, shared=shared, exchange=self.exchange,
                                           costs=costs, divided=self.divided_bases)
        except Exception as exc:
            shared, groups, rank_groups = False, {}, ()
            error = f'{type(exc).__name__}: {exc}'
        records = self.comm.allgather((int(representations), groups, error, rank_groups))
        errors = [f'rank {rank}: {record[2]}' for rank, record in enumerate(records) if record[2]]
        if errors or len({record[0] for record in records}) != 1:
            self.failed = True
            raise MPISCFError('sector configuration failed: ' + '; '.join(errors or ['representation counts differ']))
        if len({record[3] for record in records}) != 1:
            # Every rank places the sectors from the same sizes; a rank that placed them otherwise would wait in
            # collectives of other sectors.
            self.failed = True
            raise MPISCFError('sector configuration failed: the ranks placed the sectors differently')
        self.representation_count = int(representations)
        self.device_groups = groups
        self.owned_sectors = tuple(groups)
        self.shared = shared
        self.topology = tuple(record[1] for record in records)
        # The ranks of every sector (one rank each unless they share it) and the first of them, which reports its
        # Ritz values (the solve command checks every sector against it).
        self.rank_groups = rank_groups
        self.sector_owners = tuple(group[0] for group in self.rank_groups)
        if shared and int(representations) > 1:
            # Every rank joins the communicator of its sector's group; no collective of an eigensolver
            # crosses the groups after this.
            self.sector_comm = self.comm.Split(color=self.owned_sectors[0], key=self.rank)
        else:
            self.sector_comm = self.comm
        return groups

    def execute(self, solver, command, payload=None, *, density_builder=None, channel=None):
        if self.rank != self.root:
            raise MPISCFError('only root may dispatch SCF commands')
        if self.failed or self.stopped:
            raise MPISCFError('MPI sector context is no longer active')
        if channel is None:
            channel = self._active_channel(solver)
        self.comm.bcast((command, payload, channel), root=self.root)
        return self._execute_local(solver, command, payload, density_builder, channel)

    def worker_loop(self, solver, density_builder):
        if self.rank == self.root:
            raise MPISCFError('root must run SCF, not the worker loop')
        try:
            while True:
                message = self.comm.bcast(None, root=self.root)
                # (command, payload, spin channel); a command without a channel names none.
                command, payload = message[0], message[1]
                channel = message[2] if len(message) > 2 else None
                if command == 'stop':
                    self.stopped = True
                    return
                self._execute_local(solver, command, payload, density_builder, channel)
        finally:
            solver.restore_memory_allocator()

    def stop_workers(self):
        if self.rank != self.root:
            raise MPISCFError('only root may stop workers')
        if not self.failed and not self.stopped:
            self.comm.bcast(('stop', None, None), root=self.root)
            self.stopped = True

    @staticmethod
    def _active_channel(solver):
        """The spin channel the root's solver serves now, ``None`` without spin channels."""
        if getattr(solver, '_spin_channels', None) is None:
            return None
        return int(solver._active_spin_channel)

    @staticmethod
    def _select_channel(solver, channel):
        """Put spin channel ``channel`` in place in ``solver``; return the channel to put back (or None)."""
        if channel is None or solver is None:
            return None
        if getattr(solver, '_spin_channels', None) is None:
            solver.for_spin_channel(int(channel))
        previous = int(solver._active_spin_channel)
        solver._activate_spin_channel(int(channel))
        return previous

    def update_potential(self, solver, potential):
        from ..SCF.symmetry_fields import SymmetryScalarField
        if isinstance(potential, SymmetryScalarField):
            if potential.reduction is not solver.decomposition.reduction:
                raise ValueError('local potential uses a different symmetry map')
            values = potential.values
        else:
            values = solver.decomposition.invariant_wedge_values(potential)
        self.execute(solver, 'potential', np.ascontiguousarray(values, dtype=np.float64))

    def _execute_local(self, solver, command, payload, density_builder, channel=None):
        started = perf_counter()
        previous = self._select_channel(solver, channel)
        try:
            return self._execute_counted(solver, command, payload, density_builder)
        finally:
            if previous is not None:
                solver._activate_spin_channel(previous)
            self.command_seconds[command] = self.command_seconds.get(command, 0.0) + perf_counter() - started

    def _execute_counted(self, solver, command, payload, density_builder):
        self.command_counts[command] = self.command_counts.get(command, 0) + 1
        error, value, packed, signature = None, None, None, None
        try:
            value = self._apply(solver, command, payload, density_builder)
            if command == 'solve':
                # Strip both orbital arrays AND saved eigensolver states, and the record of a
                # hand-over (a device array); its plain summary, handover_step, travels.
                packed = {sector: replace(result, vectors=None, state=None, handover=None)
                          for sector, result in value.items()}
            elif command == 'density':
                value = np.ascontiguousarray(value, dtype=np.float64)
                if value.shape != (solver.decomposition.wedge_size,) or not np.all(np.isfinite(value)):
                    raise ValueError('local density must be a finite physical wedge field')
            elif command in ('forces', 'spin_forces'):
                value = np.ascontiguousarray(value, dtype=np.float64)
                if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
                    raise ValueError('local force terms must be finite, three per projector')
                # What the rows of this rank belong to, compared below before they are summed.
                signature = force_term_signature(value, getattr(solver, 'projector_force_labels', None))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            # The error text alone travels to the other ranks: where it was raised is printed here, by the rank
            # that raised it (the MPISCFError below would hide it).
            import sys
            import traceback

            print(f'MPI rank {self.rank}: {command} failed:\n{traceback.format_exc()}', file=sys.stderr, flush=True)
        records = self.comm.allgather((error, signature))
        errors = [record[0] for record in records]
        if command in ('forces', 'spin_forces') and not any(item is not None for item in errors):
            # Rows of other projectors (a rank that compressed its projectors to other atoms, or none) would be
            # summed as if they were the same, and rows of another count would leave the sum waiting: every rank
            # finds the same difference here and stops before the sum.
            errors = force_signature_mismatches([record[1] for record in records])
        if any(item is not None for item in errors):
            self.failed = True
            details = '; '.join(f'rank {rank}: {item}' for rank, item in enumerate(errors) if item)
            failure = MPISCFError(f'{command} failed: {details}')
            try:
                self._release_joined(density_builder)
            except MPISCFError as released:
                # The failure that every rank reports is the one to raise.
                raise failure from released
            raise failure
        if command == 'solve' and self.shared and self.representation_count == 1:
            # Every rank solved the shared sector together: the Ritz values are those of all of them.
            return value if self.rank == self.root else None
        if command == 'solve':
            if self.shared:
                # The ranks of a group hold the same Ritz values: its first rank alone reports them (the
                # density then sums the columns of every rank of the group once, and a basis that every rank
                # of the group holds whole on its first rank alone: whole_basis_sectors).
                packed = {sector: result for sector, result in packed.items()
                          if self.sector_owners[sector] == self.rank}
            records = self.comm.allgather(packed)
            if self.rank != self.root:
                return None
            merged = {}
            for rank, record in enumerate(records):
                for sector, result in record.items():
                    # The rank that configure placed the sector on (the first of its group where ranks share it).
                    owner = self.sector_owners[sector]
                    if owner != rank or sector in merged:
                        raise MPISCFError('sector result has duplicate or incorrect ownership')
                    merged[sector] = value[sector] if rank == self.rank else result
            if set(merged) != set(payload['representations']):
                raise MPISCFError('sector solve omitted a requested representation')
            return merged
        if command == 'density':
            total = np.empty_like(value)
            self.comm.Allreduce(value, total)
            # The pools that the builder left to their device threads (_apply) are empty before this
            # rank goes on: the root to its Hartree solve, a worker to the next command.
            self._release_joined(density_builder)
            return total
        if command == 'sector_margin':
            # Every sector takes the least share of the ranks that solve it.
            records = self.comm.allgather((tuple(self.owned_sectors), value))
            if self.rank != self.root:
                return None
            fractions = {}
            for sectors, fraction in records:
                for sector in sectors:
                    fractions[sector] = min(fractions.get(sector, 1.0), fraction)
            return fractions
        if command in ('forces', 'spin_forces'):
            # One row per projector on every rank: the ranks hold the same atoms and projectors (checked above).
            total = np.empty_like(value)
            self.comm.Allreduce(value, total)
            self._observe(command)
            return total
        return value

    def _observe(self, command):
        """Call ``command_observer`` after ``command``; what it raises is printed, not raised (no rank may leave
        the protocol alone)."""
        observer = self.command_observer
        if observer is None:
            return
        try:
            observer(command, self.command_counts.get(command, 0))
        except Exception as exc:
            import sys

            print(f'MPI rank {self.rank}: the observation after {command} failed: {type(exc).__name__}: {exc}',
                  file=sys.stderr, flush=True)

    def _release_joined(self, density_builder):
        """Wait for the pool release of ``density_builder`` where it left that to its caller.

        A release that failed in a device thread is raised here, behind the collective calls of the
        command, and so by this rank alone: as MPISCFError, with the context marked failed, so that a
        root does not send 'stop' into a job that its launcher is about to abort. The other ranks have
        their densities and are not told; they wait in their next collective call until that Abort.
        With PARSEC_CUPY_DENSITY_RELEASE=serial the release runs inside the builder, and its failure
        is reported through the command by every rank, as before.
        """
        joined = getattr(density_builder, 'release_joined', None)
        if not callable(joined):
            return
        try:
            joined()
        except Exception as exc:
            self.failed = True
            raise MPISCFError(f'density pool release failed: rank {self.rank}: {type(exc).__name__}: {exc}') from exc

    def whole_basis_sectors(self, basis_of):
        """The sectors of its group that this rank holds whole and leaves out of the sums over the ranks.

        The density and force commands add up what every rank computes from its sectors.  The ranks of a group
        that share a sector (``shared``) each hold a share of the columns of its basis where the first solve
        divided them (``Eigensolvers/mpi_shared_basis.SharedBasis``: CHEBFF, ``run_distributed_chebff``), and each
        adds its own columns.  Any other first solve (CHEBDAV, the default ``Eigensolver``) leaves every rank of
        the group with the whole basis, which it solved alone: the group's first rank, which alone reports the
        sector's Ritz values (the solve command), adds the sector once, and the other ranks add nothing for it.
        Summed by every rank of the group, the density and the force terms of such a sector counted as many times
        as the group has ranks (A100, 2026-11-18: displaced Si8 on 2 x 4 ranks twice the density, -25.7 Ry).
        ``basis_of(sector)`` is this rank's basis of a sector it owns (None where it holds none).
        """
        if not self.shared:
            return frozenset()
        from ..Eigensolvers.mpi_shared_basis import SharedBasis

        return frozenset(sector for sector in self.owned_sectors
                         if self.sector_owners[sector] != self.rank
                         and not isinstance(basis_of(sector), SharedBasis))

    def _counted_occupations(self, solvers, representations, occupations):
        """``occupations`` with those of the sectors that this rank holds whole and does not report set to zero
        (:meth:`whole_basis_sectors`; ``solvers`` are the sector solvers of the channel of the selection)."""

        def saved(sector):
            state = getattr(solvers[sector], 'device_state', None)
            return getattr(getattr(state, 'subspace', None), 'vectors', None)

        skipped = self.whole_basis_sectors(saved)
        if not skipped:
            return occupations
        return np.where(np.isin(np.asarray(representations), sorted(skipped)), 0.0,
                        np.asarray(occupations, dtype=np.float64))

    def _apply(self, solver, command, payload, density_builder):
        if command.startswith('exx_'):
            # A piece of the Fock loop of a hybrid functional (SCF/hybrid_mpi.py).
            if self.exchange_service is None:
                raise RuntimeError(f'rank {self.rank} has no exchange service for {command}')
            return self.exchange_service.handle(command, payload)
        if command == 'reset':
            solver._reset_local()
            solver._mpi_latest_results = {}
            solver._mpi_retained_results = {}
        elif command == 'configure_memory':
            solver._configure_large_problem_allocator(payload)
        elif command == 'sector_margin':
            # The share of the margin of the first counts that the devices of this rank hold.
            return float(solver._margin_fraction_local(*payload))
        elif command == 'potential':
            from ..SCF.symmetry_fields import SymmetryScalarField
            solver._update_local_potential(SymmetryScalarField(solver.decomposition.reduction, payload))
        elif command == 'solve':
            requested = tuple(index for index in payload['representations'] if index in self.device_groups)
            latest = solver._mpi_latest_results
            # The orbitals that a hybrid's exchange reads: those of the last solve of every sector,
            # kept past the density and a new SCF loop (as the views of one process are).
            retained = solver._mpi_retained_results if self.retain_orbitals else {}
            for index in requested:
                latest.pop(index, None)
                retained.pop(index, None)
            if self.rank != self.root:
                solver._sector_counts = list(payload['counts'])
            result = solver._run_local_sector_jobs(requested, payload['counts'], payload['settings'],
                                                   reset=payload['reset'])
            latest.update(result)
            retained.update(result)
            if self.rank != self.root:
                # The root counts every sector from the records it gathers; a worker its own.
                record = getattr(solver, '_record_handover', None)
                if callable(record):
                    for sector, item in result.items():
                        record(sector, item)
            return result
        elif command == 'trim':
            for sector, count in payload:
                if sector in self.device_groups:
                    solver._solvers[sector].truncate_state(count)
                solver._sector_counts[sector] = count
        elif command == 'density':
            # Keep the pre-trim result snapshots through density construction:
            # the selected columns can exceed the NEXT iteration's active set.
            representations, columns, occupations, volume = payload
            results = [solver._mpi_latest_results.get(index, SimpleNamespace(vectors=None))
                       for index in range(self.representation_count)]
            # A sector of a group that this rank holds whole is added by the group's first rank alone.
            skipped = self.whole_basis_sectors(lambda sector: results[sector].vectors)
            owned = np.isin(representations, [sector for sector in self.owned_sectors if sector not in skipped])
            orbitals = solver._pack_selected_wedge_vectors(results, representations[owned], columns[owned])
            # The pool release after the density (PARSEC_CUPY_DENSITY_RELEASE) runs on the device threads
            # beside the collective calls of this command, which touch no device, and is waited for
            # behind them (_execute_counted) instead of in the builder.
            if hasattr(density_builder, 'caller_joins_release'):
                density_builder.caller_joins_release = True
            local_density = density_builder(orbitals, occupations[owned], volume)
            from ..SCF.symmetry_fields import SymmetryScalarField
            values = (local_density.values if isinstance(local_density, SymmetryScalarField)
                      else solver.decomposition.invariant_wedge_values(local_density))
            solver._mpi_latest_results.clear()
            return values
        elif command == 'new_scf_loop':
            # The SCF loop of the next structure continues from the saved subspaces.
            solver._start_new_scf_loop_local()
        elif command == 'geometry':
            # Every rank rebuilds what it holds for the moved atoms: all of them the projectors of
            # their sectors, the root also the fields of the SCF loop.
            handler = getattr(solver, 'geometry_handler', None)
            if handler is None:
                raise RuntimeError('this rank was prepared without a geometry handler')
            handler(np.asarray(payload, dtype=np.float64))
        elif command == 'forces':
            # Nonlocal force terms of the sectors of this rank, from the states its solvers hold (a sector of a
            # group that it holds whole is added by the group's first rank alone).
            representations, columns, occupations = payload
            occupations = self._counted_occupations(solver._solvers, representations, occupations)
            return solver._projector_force_terms_local(representations, columns, occupations)
        elif command == 'spin_forces':
            # The same for both spin channels of a spin-polarized loop, summed.
            selections = []
            for index, (representations, columns, occupations) in enumerate(payload):
                solvers = (solver._solvers if getattr(solver, '_spin_channels', None) is None
                           else solver._channel_value(index, '_solvers'))
                selections.append((representations, columns,
                                   self._counted_occupations(solvers, representations, occupations)))
            return solver._spin_projector_force_terms_local(tuple(selections))
        else:
            raise ValueError(f'unknown MPI SCF command: {command}')


class MPISymmetryDensityBuilder:
    """Root SCF density hook; reduce physical wedge densities, not orbitals."""
    def __init__(self, local_builder, context, solver):
        self.local_builder, self.context, self.solver = local_builder, context, solver

    def __call__(self, wavefunctions, occupations, volume_element):
        from ..Eigensolvers.symmetry import CuPySymmetryOrbitals
        if not isinstance(wavefunctions, CuPySymmetryOrbitals):
            raise TypeError('MPI SCF requires lazy symmetry orbitals')
        weights = np.asarray(occupations, dtype=np.float64)
        if weights.shape != wavefunctions.representations.shape:
            raise ValueError('occupation count differs from the global selected states')
        payload = (wavefunctions.representations, wavefunctions.representation_columns,
                   weights, float(volume_element))
        # The orbitals of a spin channel are that channel's on every rank, whichever channel the
        # root's solver serves now.
        values = self.context.execute(self.solver, 'density', payload, density_builder=self.local_builder,
                                      channel=getattr(wavefunctions, 'spin_channel', None))
        reducer = getattr(self.local_builder, 'reducer', None)
        return reducer.field(values) if reducer is not None else values[self.solver.decomposition.reduction.full_to_wedge]


__all__ = ['MPISCFError', 'MPISectorContext', 'MPISymmetryDensityBuilder', 'force_signature_mismatches',
           'force_term_signature', 'plan_sector_groups', 'sector_device_groups', 'sector_placement_setting',
           'sector_rank_groups', 'sector_rank_plan']
