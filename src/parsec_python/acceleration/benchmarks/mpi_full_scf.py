"""Run a complete production SCF with persistent MPI symmetry-sector workers.

Use one MPI rank per node; --devices names CuPy indices within that rank's
Slurm-provided CUDA visibility. Rank zero alone owns SCF, Hartree and output.
The --serial-control path uses the unchanged production solver on one rank; it
keeps the solver's former routes where the launcher selects none.
A rank creates its CUDA contexts on a thread beside its static preparation;
PARSEC_OVERLAP_CUDA_CONTEXTS=0 creates them before it, as the serial control does.
The thread creates them through the CUDA driver library, outside Python's
interpreter lock; PARSEC_CUDA_CONTEXT_CREATION=runtime leaves them to CuPy's
first calls on each device, which hold the lock, as before.
No symmetry cache is read or written unless --symmetry-cache names a directory:
a run with default flags is the first calculation of its structure.
"""
from __future__ import annotations

import time

_MODULE_STARTED = time.perf_counter()

import argparse
import ctypes
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Settings of the serial control wherever the launcher names none: the routes
# the solver took before the measured ones became its defaults. Other runs are
# compared with the control, so it must not move with them. DSYRK is required
# here, where it used to fall back to one full product. The switches of a
# shared sector basis have no entry: the tall blocks of its Ritz step, the
# bytes of the slabs of that step, the device that estimates the condition
# number beside its small solve and the row count from which its devices read
# tiles. Without a sector context nothing is shared. The slab budget
# (PARSEC_CUPY_STREAMING_RITZ_BYTES) is the one-array route's as well and is
# left to the launcher for that reason too.
# The ionic sums give the same fields bit for bit on one device as on all. The
# control has no sector context and so sums on one by default; the entry keeps
# its preparation what it was should that default come to take more. The
# slot-major stencil is kept likewise, whatever the sectors of a device and the
# devices of a sector: a run whose sectors pack affine tiles is
# then compared with a control that reads the stencil as before. The control
# solves symmetry sectors and records filter graphs like any run, so it also
# keeps the sector stencils reduced from the full-grid matrix, the former
# symmetry maps and one graph per block of a plan: a defect of the direct
# stencil builder, of the fast maps or of the shared graphs would otherwise be
# in both sides of a comparison. PARSEC_NATIVE_SECTOR_ASSEMBLY, which chooses
# the reduction of that matrix, stays the launcher's, and the filter stays
# FP64: its former default, the FP32 later filter, is no reference.
# A control on two or three devices solves several sectors of a device one
# after another like any run: their unused pool blocks stay until the density
# step, as before.
# Where the launcher leaves the storage of the sector states or their
# allocator to the solver, the control decides by the former rule (the host
# spill and the direct allocator from half of a device in sector vectors): the
# rule that has replaced it counts one array per sector, and the control takes
# two in its Ritz step (PARSEC_CUPY_STREAMING_RITZ), so that count would keep
# on a device what the control cannot hold there. A control that spills
# downloads a basis by the former call, so that the download in the order of
# the array is not in both sides of a comparison with a run that spills.
# Neither is read where the launcher names both values, as the measured ones
# do. The contexts of the control are CuPy's: without the overlap they are in
# any case, and a launcher that asks a control for the overlap gets it with
# them, without a call into the driver library.
# No entry for the sphere: the radius is the input's, by its line or by the
# default rule of the parser, and with the Hartree tolerance it is the
# calculation that both sides share, as the boundary plan is. The report of
# the sphere (PARSEC_DOMAIN_REPORT) is no route of the solver and stays the
# launcher's; its seconds after the SCF are recorded apart
# (result.domain.after_scf). The turns of four devices on the way to their
# rows and the slab limit of two belong to a shared basis and have no entry
# either.
# The Hartree boundary of the control is that of the run: the plan made from
# the geometry (order, atomic tail) is the physics both sides must share, and
# PARSEC_HARTREE_BOUNDARY with its tolerance and tail switches stays the
# launcher's. The kernels that evaluate it do not: an engaged plan runs on the
# symmetry wedge, once per unique exterior point, with the tail from a device
# kernel, and a control that took the same kernels would be the same Hartree
# computation bit for bit. The control keeps the full-grid kernels with the
# tail in rows, and takes the values of the tail from host threads
# (PARSEC_HARTREE_ATOMIC_TAIL_VALUES): a run evaluates them with one device
# kernel whichever builder adds them, and with that kernel the rows of the
# control would hold the values of the run. What both sides still share is
# the plan, the moments of the point charges and the exterior points that the
# native builder exports to both sets of kernels.
# A symmetry cache is outside these settings: its keys name the inputs of the
# maps and stencils and not the route that built them, so a control that reads
# an entry takes what the run that wrote it built (_serial_control_notes). No
# run reads or writes one unless --symmetry-cache names its directory.
# The control also keeps the full collection after every density and the
# wedge sums and Anderson step that allocate their own arrays, so that the
# host work of a step is not the same code on both sides of a comparison, and
# the grid builder of the reference package, for the same reason. On several
# devices it empties their pools after a density one after another, in the
# thread that built the density, as before.
# The slabs of its projection keep the widths they had: a run cuts the slabs
# of both Gram matrices at multiples of 64 columns, which sums them from other
# pieces. The overlap of the control has no slabs: it is DSYRK's
# (PARSEC_CUPY_RITZ_SYRK=on, the first setting below) unless its launcher
# names another policy. Its projection is formed with the slabs of
# PARSEC_CUPY_RITZ_GRAM_SLABS, and with those of the streaming budget where
# its launcher asks it for one array, so the control reads the multiple on
# either route.
# No entry for the release that a shared basis no longer takes after a density
# (PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE): only a sector context gives an
# operator the devices of a group, the control has none, and so no basis of it
# is shared on any number of devices. Its density step empties the pool of
# every device that holds a sector, under either value of that switch.
_SERIAL_CONTROL_SETTINGS = {
    'PARSEC_CUPY_RITZ_SYRK': 'on',
    'PARSEC_CUPY_RITZ_DENSE_BACKEND': 'host',
    # The ionic fields in line and the CUDA contexts before the preparation, so
    # that a run that builds the fields on their own thread is compared with
    # fields that were not.
    'PARSEC_OVERLAP_IONIC_SETUP': '0',
    'PARSEC_OVERLAP_CUDA_CONTEXTS': '0',
    'PARSEC_CUDA_CONTEXT_CREATION': 'runtime',
    'PARSEC_CUPY_ROTATE_COLUMN_COPY': '0',
    'PARSEC_CUPY_STREAMING_RITZ': '0',
    'PARSEC_CUPY_STREAMING_RITZ_AUTO_FRACTION': '0.8',
    'PARSEC_CUPY_SECTOR_POOL_RELEASE': '0',
    'PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION': '0.5',
    'PARSEC_CUPY_SPILL_DOWNLOAD_ORDER': 'C',
    'PARSEC_HARTREE_DEVICE': 'off',
    'PARSEC_IONIC_GPU_COUNT': '1',
    'PARSEC_CUPY_IMPLICIT_TILE': '0',
    'PARSEC_SECTOR_STENCIL': 'csr',
    'PARSEC_SYMMETRY_FAST_MAPS': '0',
    'PARSEC_CUPY_FILTER_GRAPH_REUSE': '0',
    'PARSEC_CUPY_DENSITY_COLLECTION': 'full',
    'PARSEC_CUPY_DENSITY_RELEASE': 'serial',
    'PARSEC_CUPY_RITZ_GRAM_MULTIPLE': '1',
    'PARSEC_SYMMETRY_SCF_BUFFERS': '0',
    'PARSEC_FAST_GRID': '0',
    'PARSEC_HARTREE_BOUNDARY_KERNEL': 'full',
    'PARSEC_HARTREE_ATOMIC_TAIL_VALUES': 'host',
}


def _keep_former_routes(environment):
    """Give ``environment`` the serial control's settings where it names none."""
    for name, value in _SERIAL_CONTROL_SETTINGS.items():
        environment.setdefault(name, value)


def _serial_control_notes(cache):
    """Lines for the log of a serial control: its settings, and what a symmetry cache ``cache`` does to them."""
    lines = ['Serial control routes: ' + ' '.join(
        f'{name}={os.environ.get(name)}' for name in _SERIAL_CONTROL_SETTINGS)]
    if cache is not None:
        lines.append(f'WARNING: the serial control reads the symmetry cache {cache}. An entry that another run wrote '
                     'there gives it the symmetry maps and sector stencils of that run, whatever PARSEC_SECTOR_STENCIL '
                     'and PARSEC_SYMMETRY_FAST_MAPS say here; a control without --symmetry-cache reads none and stays '
                     'independent.')
    return lines


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if hasattr(value, 'item'):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _environment():
    names = ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES',
             'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'SLURM_JOB_ID', 'SLURM_JOB_NODELIST',
             'SLURM_NNODES', 'SLURM_NTASKS', 'SLURM_PROCID', 'SLURM_LOCALID',
             'SLURM_CPUS_PER_TASK', 'MPICH_GPU_SUPPORT_ENABLED', 'MPICH_OFI_NIC_POLICY',
             'MPICH_ASYNC_PROGRESS')
    result = {key: os.environ.get(key) for key in names}
    result.update({key: value for key, value in os.environ.items() if key.startswith('PARSEC_')})
    return result


def _source_provenance():
    source = Path(__file__).resolve().parents[2]
    root = source.parents[1]
    hashes = {str(path.relative_to(root)).replace('\\', '/'): _digest(path)
              for path in sorted(source.rglob('*.py'))}
    def git(*arguments):
        try:
            return subprocess.run(['git', '-C', str(root), *arguments], capture_output=True,
                                  check=True, timeout=15).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    commit = git('rev-parse', 'HEAD')
    diff = git('diff', '--binary', 'HEAD', '--', 'src/parsec_python')
    status = git('status', '--short', '--', 'src/parsec_python')
    return dict(git_commit=None if commit is None else commit.decode().strip(),
                git_source_status=None if status is None else status.decode().strip(),
                git_tracked_source_diff_sha256=None if diff is None else hashlib.sha256(diff).hexdigest(),
                source_sha256=hashes)


def _parse_devices(text):
    try:
        devices = tuple(int(item.strip()) for item in text.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('devices must be comma-separated integer CuPy indices') from exc
    if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError('devices must be nonempty, distinct and nonnegative')
    return devices


def _require_visible(cp, devices):
    """Refuse a device index that the process cannot see."""
    count = cp.cuda.runtime.getDeviceCount()
    if max(devices) >= count:
        raise ValueError(f'requested devices {devices}; only {count} visible CuPy devices')


def _driver_contexts_requested():
    """PARSEC_CUDA_CONTEXT_CREATION: driver (the default) or runtime."""
    value = os.environ.get('PARSEC_CUDA_CONTEXT_CREATION', 'driver').strip().lower()
    if value not in {'driver', 'runtime'}:
        raise ValueError('PARSEC_CUDA_CONTEXT_CREATION must be driver or runtime')
    return value == 'driver'


class _DriverContexts:
    """Create primary CUDA contexts through the driver library, outside the interpreter lock.

    CuPy creates the context of a device inside its first runtime call that needs one and holds the
    interpreter lock for that call: a thread that creates contexts this way stops every other Python thread
    meanwhile, the one that prepares beside it included. ctypes releases the lock around a foreign call.
    cuDevicePrimaryCtxRetain creates the primary context of a device, which is the one the CUDA runtime and
    so CuPy use, or takes a reference to it where it exists; the CuPy calls that follow find it. The
    reference is kept for the life of the process, as the runtime keeps its own.

    The driver and the runtime number the visible devices alike. That is checked and not assumed: a context
    is created here only where the driver gives the device of an index the PCI address that CuPy reports
    for it. Where the library cannot be loaded, a call fails or the addresses differ, nothing is created
    here and CuPy creates the context as before, with its own errors. ``devices`` lists the devices whose
    context was created or found here and ``seconds`` is the time inside the driver calls.
    """

    def __init__(self, library=None):
        self.devices, self.seconds = [], 0.0
        try:
            self.library = library or ctypes.CDLL('nvcuda.dll' if os.name == 'nt' else 'libcuda.so.1')
        except OSError:
            self.library = None

    def _call(self, name, *arguments):
        """Whether the driver function ``name`` returned success."""
        started = time.perf_counter()
        try:
            return self.library is not None and getattr(self.library, name)(*arguments) == 0
        except (AttributeError, OSError):
            return False
        finally:
            self.seconds += time.perf_counter()-started

    def start(self):
        """Initialize the driver, which the first runtime call of the process would otherwise do."""
        return self._call('cuInit', ctypes.c_uint(0))

    def create(self, cp, device):
        """Create the context of ``device`` unless the driver knows another device by that index."""
        handle, context, address = ctypes.c_int(), ctypes.c_void_p(), ctypes.create_string_buffer(32)
        if not (self._call('cuDeviceGet', ctypes.byref(handle), ctypes.c_int(device))
                and self._call('cuDeviceGetPCIBusId', address, ctypes.c_int(len(address)), handle)):
            return False
        if address.value.decode('ascii', 'replace').lower() != str(cp.cuda.Device(device).pci_bus_id).lower():
            return False
        if not self._call('cuDevicePrimaryCtxRetain', ctypes.byref(context), handle):
            return False
        self.devices.append(device)
        return True


def _create_contexts(cp, devices, driver=None):
    """Create the CUDA context of every device in turn and describe the devices.

    A ``driver`` (:class:`_DriverContexts`) creates each of them first, outside the interpreter lock; the
    CuPy calls here then find it.
    """
    if driver is not None:
        driver.start()
    _require_visible(cp, devices)
    records = []
    for device in devices:
        if driver is not None:
            driver.create(cp, device)
        with cp.cuda.Device(device):
            free, total = cp.cuda.runtime.memGetInfo()
            properties = cp.cuda.runtime.getDeviceProperties(device)
            records.append(dict(index=device, pci_bus_id=cp.cuda.Device().pci_bus_id,
                                name=_jsonable(properties['name']), total_bytes=int(total),
                                initial_free_bytes=int(free)))
    return records


def _initialize_devices(devices):
    # This full-SCF protocol sends host fields only, so GPU context creation
    # need not precede MPI.Init. Keep CUDA cold-start cost in preparation.
    import cupy as cp
    records = _create_contexts(cp, devices)
    cp.cuda.Device(devices[0]).use()
    os.environ['PARSEC_CUPY_DEVICES'] = ','.join(map(str, devices))
    return cp, records


def _context_overlap_requested():
    """PARSEC_OVERLAP_CUDA_CONTEXTS: on unless 0, false, no or off."""
    return os.environ.get('PARSEC_OVERLAP_CUDA_CONTEXTS', '1').strip().lower() not in {'0', 'false', 'no', 'off'}


class _ContextCreation:
    """Create the CUDA contexts on a thread while the caller prepares host objects.

    Grid, pseudopotentials, finite differences and projectors need no device. A thread that reaches a device
    whose context is being created waits for it inside the CUDA runtime, and one that reaches a device first
    creates the context itself: the GPU ionic sums of the driver can, on their own thread. The memory
    observation the thread here takes can then include their arrays. The thread here selects devices and
    reads their memory and properties: it makes no MPI call and takes no cuBLAS or cuSOLVER handle, so its
    end leaves a graph capture valid (backends/cupy_capture.py).

    With ``through_driver`` the thread first creates each context through the CUDA driver library
    (:class:`_DriverContexts`, kept as ``driver``), which leaves the interpreter lock to the caller
    meanwhile. Without it the contexts come into being inside CuPy's calls, which hold the lock: the caller
    then runs between two devices only. A caller that selects a device through CuPy while its context is
    being created waits for it with the lock held, either way.
    """

    def __init__(self, cp, devices, through_driver=False):
        self.cp, self.devices, self.through_driver = cp, tuple(devices), bool(through_driver)
        self.records = self.sampler = self.observation = self.error = self.driver = None
        self.seconds = self.wait_seconds = 0.0
        # A thread starts with device 0 current. Any other first device is selected by the caller itself,
        # which is where the runtime may create its context. The caller then also refuses a device that the
        # process cannot see, with the words it has without the overlap, before the runtime does.
        if self.devices[0] != 0:
            _require_visible(cp, self.devices)
            cp.cuda.Device(self.devices[0]).use()
        os.environ['PARSEC_CUPY_DEVICES'] = ','.join(map(str, self.devices))
        self._thread = threading.Thread(target=self._create, name='parsec-cuda-contexts')
        self._thread.start()

    def _create(self):
        started = time.perf_counter()
        try:
            if self.through_driver:
                self.driver = _DriverContexts()
            self.records = _create_contexts(self.cp, self.devices, self.driver)
            self.seconds = time.perf_counter()-started
            # As without the overlap: nvidia-smi starts once the contexts exist, not beside their creation.
            self.sampler = _GpuMemorySampler()
            self.observation = _observe_memory(self.cp, self.devices, 'after_cuda_init_during_static_preparation')
        except BaseException as error:
            self.error = error

    def join(self):
        """Wait for the thread; return the device records or raise what stopped it."""
        started = time.perf_counter()
        self._thread.join()
        self.wait_seconds = time.perf_counter()-started
        if self.error is not None:
            raise self.error
        return self.records


def _restore_affinity(initial):
    """Give the calling thread back the CPU mask the launcher granted.

    With OMP_PROC_BIND set, the OpenMP runtime pins the initial thread to its
    first place. Python threads started later (one per symmetry sector and
    per filter device) inherit that one-core mask and then share it for host
    LAPACK and CUDA launches. OpenMP teams keep their own places.
    """
    if initial is None or set(os.sched_getaffinity(0)) == set(initial):
        return False
    os.sched_setaffinity(0, initial)
    return True


class _GpuMemorySampler:
    """Continuously sample used memory of every GPU on this node.

    The endpoint observations above miss transient peaks inside an eigensolve.
    A separate ``nvidia-smi`` process polls the driver, so sampling needs no
    CUDA call or interpreter lock in this process. Values include the CUDA
    context and blocks cached by CuPy's pool, i.e. what must fit on the card.
    """

    def __init__(self, interval_ms=200):
        import tempfile
        self.interval_ms = int(interval_ms)
        self.output = tempfile.TemporaryFile(mode='w+')
        try:
            self.process = subprocess.Popen(
                ['nvidia-smi', '--query-gpu=pci.bus_id,memory.used', '--format=csv,noheader,nounits',
                 f'--loop-ms={self.interval_ms}'],
                stdout=self.output, stderr=subprocess.DEVNULL)
        except OSError:
            self.process = None
        else:
            import atexit
            atexit.register(self.process.kill)

    def stop(self, device_records):
        """Return per-device peaks, keyed like ``device_records`` when the bus id matches."""
        if self.process is None:
            return dict(available=False)
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.output.seek(0)
        peaks, counts = {}, {}
        for line in self.output:
            bus, _, used = line.strip().rpartition(',')
            try:
                value = int(used)
            except ValueError:
                continue
            key = bus.strip().lower()[-12:]
            peaks[key] = max(peaks.get(key, 0), value)
            counts[key] = counts.get(key, 0) + 1
        self.output.close()
        selected = {str(item['index']): peaks.get(str(item['pci_bus_id']).lower()[-12:]) for item in device_records}
        return dict(available=bool(peaks), interval_ms=self.interval_ms, samples_per_device=min(counts.values(), default=0),
                    peak_used_mib_by_bus=peaks, peak_used_mib_by_selected_device=selected,
                    peak_used_mib=max(peaks.values(), default=None))


def _synchronize(cp, devices):
    for device in devices:
        with cp.cuda.Device(device):
            cp.cuda.runtime.deviceSynchronize()


def _observe_memory(cp, devices, label):
    """Endpoint observations, not a claim to continuous GPU allocation peaks."""
    try:
        import resource
        scale = 1 if sys.platform == 'darwin' else 1024
        rss_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)
    except ImportError:
        rss_peak = None
    current_rss = None
    try:
        current_rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, AttributeError):
        pass
    gpu = []
    for device in devices:
        with cp.cuda.Device(device):
            free, total = cp.cuda.runtime.memGetInfo()
            pool = cp.get_default_memory_pool()
            gpu.append(dict(index=device, device_used_bytes=int(total-free),
                            default_cupy_pool_reserved_bytes=int(pool.total_bytes()),
                            default_cupy_pool_used_bytes=int(pool.used_bytes())))
    return dict(phase=label, rss_current_bytes=current_rss, process_rss_high_water_bytes=rss_peak, devices=gpu)


def _prepare_on_devices(cp, devices, prepare, memory):
    """Create the CUDA contexts and run ``prepare()``, beside each other unless switched off.

    Returns cp, what ``prepare`` returned, the device records, the memory sampler, and of the context
    creation its seconds, whether it ran beside the preparation, how long its join waited, the devices
    whose context the thread created through the driver library and the seconds inside those calls
    (PARSEC_CUDA_CONTEXT_CREATION). Without the overlap the contexts exist before ``prepare`` is called,
    as they used to, and CuPy creates them: nothing runs beside them that the interpreter lock could stop.
    """
    started = time.perf_counter()
    # Read in every run, so that a value it does not know stops a run that does not use it either.
    through_driver = _driver_contexts_requested()
    if not _context_overlap_requested():
        cp, records = _initialize_devices(devices)
        seconds = time.perf_counter()-started
        sampler = _GpuMemorySampler()
        memory.append(_observe_memory(cp, devices, 'after_cuda_init_before_static_preparation'))
        return cp, prepare(), records, sampler, dict(seconds=seconds, overlapped=False, join_wait_seconds=0.0,
                                                     driver_devices=[], driver_seconds=0.0)
    contexts = _ContextCreation(cp, devices, through_driver)
    try:
        prepared = prepare()
    finally:
        # Also after a preparation that failed: a device that cannot be used is the cause to report.
        records = contexts.join()
    memory.append(contexts.observation)
    created = contexts.driver
    return cp, prepared, records, contexts.sampler, dict(
        seconds=contexts.seconds, overlapped=True, join_wait_seconds=contexts.wait_seconds,
        driver_devices=[] if created is None else list(created.devices),
        driver_seconds=0.0 if created is None else created.seconds)


def _parsed_domain(translation):
    """Sphere radius and Hartree boundary tolerance as this rank parsed them.

    Their bit patterns, and the texts of the default rule where it chose
    them: every rank runs the rule on its own, from the input and the
    pseudopotential bytes that the ranks have just compared.
    """
    problem, choice = translation.problem, getattr(translation, 'domain', None)
    tolerance = problem.hartree.boundary_tolerance
    return (float(problem.grid.radius).hex(), None if tolerance is None else float(tolerance).hex(),
            None if choice is None else (choice.radius_text, choice.tolerance_text))


def _require_same_domain(parsed):
    """Stop unless every rank resolved the domain of the first, before any of them prepares a grid."""
    different = [rank for rank, item in enumerate(parsed) if item != parsed[0]]
    if different:
        raise ValueError('the sphere radius or the Hartree boundary tolerance differs across ranks: rank 0 has '
                         f'{parsed[0]}, rank {different[0]} has {parsed[different[0]]}; give Boundary_Sphere_Radius '
                         'and Hartree_Boundary_Tolerance in the input')


def _hartree_boundary_record(system, result):
    """The Hartree boundary a run had, for whoever compares it with another run.

    The environment switches of the boundary change these values without
    changing the input, so two runs are the same calculation only where the
    order, the tolerance and the tail agree.  ``kernel`` says which builder
    evaluated them and ``atomic_tail_values`` what evaluated the tail; both
    may differ: the serial control keeps the full-grid kernels and the host
    threads.  ``None`` for a system prepared without a plan.  What these
    boundary values are estimated to leave in the energy is no part of this
    record: it is in ``result.domain``.
    """
    plan = getattr(system, 'hartree_boundary', None)
    if plan is None:
        return None
    details = dict(result.backend.details)
    return dict(multipole_order=int(plan.order), solver_lpole=int(plan.minimum_order),
                tolerance_ry=plan.tolerance, atomic_tail=bool(plan.atomic_tail),
                kernel=details.get('hartree_boundary_kernel'),
                atomic_tail_values=details.get('hartree_atomic_tail_values'))


def _stack_phases(result, scf_seconds, reported_seconds, check_seconds=0.0):
    timings = result.timings
    first = float(result.history[0].diagonalization_seconds) if result.history else 0.0
    diagonalization = float(timings.diagonalization_seconds)
    density = float(timings.occupations_density_seconds)
    hartree = float(timings.hartree_seconds + timings.initial_hartree_seconds)
    phases = dict(first_iteration_diagonalization_seconds=first,
                  subsequent_diagonalization_seconds=diagonalization-first,
                  occupations_density_seconds=density, hartree_seconds=hartree,
                  other_scf_seconds=scf_seconds-diagonalization-density-hartree,
                  hartree_boundary_check_seconds=check_seconds,
                  preparation_reporting_other_program_seconds=reported_seconds-scf_seconds-check_seconds)
    if any(not math.isfinite(value) or value < -1e-7 for value in phases.values()):
        raise ValueError(f'non-disjoint wall-time stack: {phases}')
    if not math.isclose(sum(phases.values()), reported_seconds, abs_tol=1e-7, rel_tol=1e-12):
        raise ValueError('wall-time stack does not close')
    return phases


def execute(args, cp, comm, MPI, initialization_seconds, initial_affinity=None):
    import numpy as np
    from parsec_python.Input import parse_parsec_input
    from parsec_python.cli import save_result_archive
    from parsec_python.acceleration.cli import _RunLog, _symmetry_cache_directory
    from parsec_python.acceleration.Output import AcceleratedTextReporter
    from parsec_python.acceleration.backends.cupy_capture import capture_statistics
    from parsec_python.acceleration.driver import hartree_boundary_check_seconds, prepare_single_point, run_scf
    from parsec_python.acceleration.experimental.mpi_scf import MPISectorContext, MPISymmetryDensityBuilder

    affinity_restorations = int(_restore_affinity(initial_affinity))
    rank, size = comm.rank, comm.size
    if args.serial_control and size != 1:
        raise ValueError('--serial-control requires exactly one MPI rank')
    hosts = comm.allgather(socket.gethostname())
    if len(set(hosts)) != size:
        raise ValueError('full-SCF runner requires one MPI rank per node')

    translation = parse_parsec_input(args.input, pseudopotential_directory=args.pp_dir)
    if translation.problem.periodic_cell is not None:
        # Refused here by name: the sphere and the symmetry sectors read below are those of a cluster.
        raise ValueError('the full-SCF runner divides the symmetry sectors of an isolated cluster; a periodic input '
                         '(Boundary_Conditions: bulk) is not supported: run it with parsec_python/main.py')
    symmetry = args.symmetry or ('off' if translation.ignore_symmetry else 'auto')
    # As in acceleration.cli: none unless --symmetry-cache names its directory.
    cache = _symmetry_cache_directory(args)
    output = args.output_dir.resolve()
    log_path, archive_path, timing_path = output/'parsec.out', output/'parsec_python_results.npz', output/'timing.json'
    protected = {translation.source.resolve(), *(item.path.resolve() for item in translation.problem.pseudopotentials.values())}
    if any(path in protected for path in (log_path, archive_path, timing_path)):
        raise ValueError('output would overwrite physical input')
    if any(path.exists() for path in (log_path, archive_path, timing_path)):
        raise FileExistsError('use a new output directory; previous SCF artifacts are protected')
    inputs = {str(path): _digest(path) for path in sorted(protected)}
    source = _source_provenance() if rank == 0 else None
    identical_inputs = comm.allgather(inputs)
    if any(item != inputs for item in identical_inputs):
        raise ValueError('input or pseudopotential content differs across ranks')
    _require_same_domain(comm.allgather(_parsed_domain(translation)))
    context = None if args.serial_control else MPISectorContext(comm)
    memory = []
    # As in acceleration.cli, this clock begins after imports, input parsing
    # and output-path resolution. All rank readiness/preparation waits follow.
    # The default rule of the sphere ran inside the parsing: its seconds on
    # this rank are counted. There are none for an input with a radius.
    started = time.perf_counter()-translation.domain_rule_seconds
    reporter = None
    log_manager = _RunLog(log_path if rank == 0 else None, quiet=args.quiet or rank != 0)
    with log_manager as log:
        if rank == 0:
            reporter = AcceleratedTextReporter(log.write, translation, symmetry_mode=symmetry)
            reporter.header()
            for warning in translation.warnings:
                log.write(f'WARNING: {warning}')
            if translation.output_all_states:
                log.write('BENCHMARK: Output_All_States overridden false; no full orbital materialization/archive.')
            log.write(f'Full SCF mode: {"serial production control" if args.serial_control else "MPI symmetry sectors"}; '
                      f'{size} ranks/nodes, {size*len(args.devices)} GPUs total; devices/rank={args.devices}; '
                      'MPI transport=host-fields')
            if args.serial_control:
                for line in _serial_control_notes(cache):
                    log.write(line)
            log.write()
        readiness_started = time.perf_counter()
        comm.Barrier()
        readiness_seconds = time.perf_counter()-readiness_started
        preparation_started = time.perf_counter()
        options = {} if context is None else {'mpi_context': context}
        cp, system, device_records, sampler, cuda_contexts = _prepare_on_devices(
            cp, args.devices,
            lambda: prepare_single_point(translation.problem, backend=args.backend, symmetry=symmetry,
                                         symmetry_cache_directory=cache, **options),
            memory)
        cuda_initialization_seconds = cuda_contexts['seconds']
        system.materialize_final_wavefunctions = False
        _synchronize(cp, args.devices)
        preparation_seconds = time.perf_counter()-preparation_started
        if system.backend_info.selected != 'cupy':
            raise RuntimeError('full-SCF GPU benchmark may not silently fall back to a CPU backend')
        solver = getattr(system.backend, 'symmetry_eigensolver', None)
        if context is not None and (solver is None or not context.owned_sectors):
            raise RuntimeError('MPI full SCF requires the configured symmetry-sector eigensolver')
        memory.append(_observe_memory(cp, args.devices, 'after_preparation'))
        ready_started = time.perf_counter()
        placement = comm.allgather(dict(hostname=socket.gethostname(), devices=device_records))
        physical_devices = [(row['hostname'], item['pci_bus_id']) for row in placement for item in row['devices']]
        if len(set(physical_devices)) != len(physical_devices):
            raise ValueError('ranks selected duplicate GPUs')
        comm.Barrier()
        preparation_wait_seconds = time.perf_counter()-ready_started
        # Sector and filter-device threads start lazily inside the SCF loop.
        affinity_restorations += int(_restore_affinity(initial_affinity))
        if rank == 0 and affinity_restorations:
            log.write('NOTE: OpenMP binding had pinned the calling thread; its launcher CPU mask '
                      f'({len(initial_affinity)} CPUs) was restored for Python worker threads.')
        prepared_at = time.perf_counter()
        result = None
        # The rank's own density builder: the root wraps it for the MPI protocol below.
        density_builder = getattr(system, 'orbital_density_builder', None)
        if rank == 0:
            if context is not None:
                system.orbital_density_builder = MPISymmetryDensityBuilder(system.orbital_density_builder, context, solver)
            reporter.setup(system)
            def iteration(item):
                reporter.iteration(item)
                memory.append(_observe_memory(cp, args.devices, f'iteration_{item.iteration}'))
            scf_started = time.perf_counter()
            try:
                result = run_scf(system, callback=iteration)
                _synchronize(cp, args.devices)
                scf_completed = time.perf_counter()
            finally:
                if context is not None:
                    context.stop_workers()
        else:
            context.worker_loop(solver, system.orbital_density_builder)
        _synchronize(cp, args.devices)
        worker_completed_at = time.perf_counter()
        memory.append(_observe_memory(cp, args.devices, 'after_scf_or_worker_loop'))
        # This join is part of reported program time, not an excluded overhead.
        comm.Barrier()
        joined_at = time.perf_counter()
        gpu_memory = sampler.stop(device_records)
        root_report = None
        if rank == 0:
            # run_scf makes the optional check of the Hartree boundary (PARSEC_HARTREE_BOUNDARY_CHECK) before
            # it returns: its seconds are a phase of their own and no part of the SCF.
            check_seconds = hartree_boundary_check_seconds(result)
            scf_seconds = scf_completed-scf_started-check_seconds
            reporter.finish(result, scf_seconds)
            reporting_completed = time.perf_counter()
            log.write(f' Pre-SCF setup/reporting wall time [sec] : {scf_started-started:11.6f}')
            log.write(f' Post-SCF finalization/reporting [sec] : {reporting_completed-scf_completed+check_seconds:11.6f}')
            reported_seconds = time.perf_counter()-started
            log.write(f' Total accelerated Python wall time [sec] : {reported_seconds:11.2f}')
            # What the reporter kept of the sphere and of the estimates it printed: a dict, and none for a box.
            domain = reporter.domain if isinstance(reporter.domain, dict) else None
            root_report = dict(converged=bool(result.converged), iterations=int(result.iterations),
                electron_count=float(result.electron_count), energies_ry=asdict(result.energies),
                fermi_level_ry=float(result.fermi_level), backend=asdict(result.backend),
                backend_statistics=asdict(result.backend_statistics), scf_timings=asdict(result.timings),
                iteration_history=[asdict(item) for item in result.history],
                reported_program_seconds=reported_seconds, scf_seconds=scf_seconds,
                pre_scf_setup_reporting_seconds=scf_started-started,
                post_scf_finalization_reporting_seconds=reporting_completed-scf_completed+check_seconds,
                hartree_boundary_check_seconds=check_seconds,
                hartree_boundary=_hartree_boundary_record(system, result),
                domain=domain,
                first_iteration_diagonalization_seconds=float(result.history[0].diagonalization_seconds) if result.history else 0.0,
                wall_time_stack_seconds=_stack_phases(result, scf_seconds, reported_seconds, check_seconds))
            archive_started = time.perf_counter()
            save_result_archive(archive_path, result, include_wavefunctions=False)
            root_report['archive_write_seconds_excluded_from_reported'] = time.perf_counter()-archive_started
            root_report['archive_sha256'] = _digest(archive_path)
            log.write(f'Result archive: {archive_path}')
            log.write(f'Text log: {log_path}')

    # Wall time of each stage of the multi-device sector route, when it ran, and of the seconds of its Gram
    # stage those that the overlap took: the rest of that stage is the projection, whose products are as wide
    # as a round or a slab of the step (0 for steps with three tall blocks, which form both matrices in one
    # call). Then the column ranges that each device held in the last filter, the tall blocks a device held in
    # the last Ritz step (None before the first), how that step cut the slabs of H X (equal or full, or
    # multiple where a step with one tall block cut rounds of whole multiples of
    # PARSEC_CUPY_RITZ_GRAM_MULTIPLE columns), how many columns its widest slab had and how many the widest
    # right-hand side of a product of its projection, a round with one tall block and a slab with two (None
    # with three tall blocks, which cut none), and the row blocks that Ritz steps with one tall block per
    # device had to take from the pool because the block of the columns had no room for them. Also the
    # device that estimated the condition number of the overlap in the last device small solve (the owner, or
    # the device that did so beside the solve; None before the first and for host solves), and the seconds
    # that such helpers spent on it: they run beside the dense stage and are no part of the stages. And the
    # order in which the devices issued the copies between them on the last way to their rows: pairs (four
    # devices, a partner per turn for the chunks that all of them send), together or queued on the one
    # stream of each (None before the first).
    shared_state = []
    for sector, operator in enumerate(getattr(solver, '_operators', None) or ()):
        group = getattr(operator, '_sector_device_group', None)
        if group is not None:
            layout = getattr(group, 'layout', None)
            shared_state.append(dict(sector=sector, devices=list(group.devices), ritz_passes=group.passes,
                                     seconds=dict(group.seconds), overlap_seconds=group.overlap_seconds,
                                     tall_blocks=group.tall_blocks,
                                     slab_cut=group.slab_cut, slab_columns=group.slab_columns,
                                     projection_columns=group.projection_columns,
                                     separate_row_blocks=group.separate_row_blocks,
                                     condition_device=group.condition_device,
                                     condition_seconds=group.condition_seconds,
                                     to_rows_copies=group.to_rows_copies,
                                     column_ranges=None if layout is None else [list(map(list, part)) for part in layout.parts]))
            # The layout of the stencil that the devices filtered with, and the wall time in which the operators
            # of the devices beside the owner were built: it lies in the first solve, outside the stages.
            shared_state[-1].update(stencil_storage=group.stencil_storage, replica_seconds=group.replica_seconds)
    local = dict(rank=rank, hostname=socket.gethostname(), devices=device_records,
        environment=_environment(), cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        launcher_cpu_affinity=initial_affinity, cpu_affinity_restorations=affinity_restorations,
        cupy=cp.__version__, numpy=np.__version__, mpi_library=MPI.Get_library_version().strip(),
        preparation_seconds=preparation_seconds, preparation_wait_seconds=preparation_wait_seconds,
        domain_rule_seconds=translation.domain_rule_seconds,
        cuda_context_initialization_seconds_in_preparation=cuda_initialization_seconds,
        cuda_context_initialization_overlapped=cuda_contexts['overlapped'],
        cuda_context_join_wait_seconds=cuda_contexts['join_wait_seconds'],
        cuda_context_driver_devices=cuda_contexts['driver_devices'],
        cuda_context_driver_seconds=cuda_contexts['driver_seconds'],
        initial_readiness_wait_seconds=readiness_seconds, ready_after_reported_start_seconds=prepared_at-started,
        worker_or_root_active_seconds=worker_completed_at-prepared_at,
        final_join_wait_seconds=joined_at-worker_completed_at,
        initialization_before_main_imports_seconds=initialization_seconds,
        external_module_wall_seconds_through_archive=time.perf_counter()-_MODULE_STARTED,
        owned_sectors=None if context is None else context.owned_sectors,
        device_groups=None if context is None else context.device_groups,
        command_counts=None if context is None else context.command_counts,
        command_seconds=None if context is None else context.command_seconds,
        density_pool_release=getattr(density_builder, 'pool_release', None),
        backend_statistics=asdict(system.backend.statistics), memory_observations=memory,
        graph_captures=capture_statistics(),
        # Per device: releases of a finished sector's pool blocks and the bytes returned; empty where none ran.
        sector_pool_release=dict(getattr(solver, 'sector_pool_releases', None) or {}),
        gpu_memory=gpu_memory, distributed_state=shared_state,
        single_array_ritz_sectors=list(getattr(solver, 'low_memory_ritz_sectors', ()) or ()),
        # Where the solver of the rank keeps the states of its sectors, which allocator serves them, and the
        # bytes per device that ``auto`` of either held against its share of the memory of the device
        # (PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION; None where it did not ask: both named, or the former rule
        # by PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION).
        orbital_sector_state_storage=getattr(solver, 'sector_state_storage', None),
        orbital_memory_allocator=getattr(solver, 'memory_allocator_policy', None),
        sector_state_fit_bytes=getattr(solver, 'state_fit_bytes', None))
    ranks = comm.gather(local, root=0)
    exit_code = None
    if rank == 0:
        exit_code = 0 if result.converged else 3
        payload = dict(kind='complete_scf_mpi_symmetry_sectors', schema_version=1,
            status='converged' if result.converged else 'not_converged', complete=True,
            serial_control=args.serial_control, ranks=size, nodes=len(placement), gpu_count=len(physical_devices),
            mpi_transport='host-fields', symmetry_cache_directory=cache,
            parameters=vars(args), input_sha256=inputs, source_provenance=source,
            rank_topology=placement, per_rank=ranks, result=root_report,
            preparation_seconds_max_rank=max(row['preparation_seconds'] for row in ranks),
            gpu_peak_used_mib_max=max((row['gpu_memory'].get('peak_used_mib') or 0 for row in ranks), default=0) or None,
            external_module_wall_seconds_max_rank=max(row['external_module_wall_seconds_through_archive'] for row in ranks),
            notes=[
                'Actual production SCF: all occupied/requested states, all symmetry sectors and physical density/Hartree updates.',
                'One MPI rank per node. Local devices are CuPy indices inside unchanged Slurm CUDA_VISIBLE_DEVICES.',
                'symmetry_cache_directory is the directory --symmetry-cache named. null, the default, is a run without a symmetry cache: no rank read or wrote one, as in the first calculation of a structure. With a directory every rank loads or builds the stencils of all sectors and may have read what an earlier run wrote.',
                'Only root executes SCF/Hartree; workers keep sector orbitals resident. No full orbital archive/materialization.',
                'Preparation follows the role: every rank builds grid, symmetry maps, projectors and its own sectors; only root also builds ionic fields, densities, the totally symmetric Poisson operator, Hartree and XC. Per-rank preparation_seconds differ accordingly.',
                'Reported program wall matches acceleration.cli boundaries: after imports/input parsing through final reporting, before archive.',
                'result.hartree_boundary is the Hartree boundary of the run: multipole order in use, Solver_Lpole, tolerance, atomic tail, the kernels that evaluated it and what evaluated the tail (null without one). The PARSEC_HARTREE_* switches change it without changing input_sha256: a comparison of two runs is one of the same calculation only where order, tolerance and tail agree. The kernels and the evaluator of the tail may differ: a serial control has the full-grid kernels and host threads.',
                'result.domain is the sphere of the run (null for a box, and for an input with a radius under PARSEC_DOMAIN_REPORT=0, which also leaves after_scf out of every run): rule_version, radius_from (input or default rule), radius_bohr, the outermost atom and the vacuum beyond it, and, where the default rule chose the radius, boundary_sphere_radius and hartree_boundary_tolerance, the values of the two input lines that give this domain again, with set_by. wall_estimate_ry is the estimate before the SCF of the energy the sphere adds, from the free atoms: an estimate and not a bound, and meaningless with extra_electrons above zero; its parts by species and the source and charge of every free-atom density are beside it. boundary holds the estimate of the energy that the Hartree boundary values of the plan in use leave: about_ry and bound_ry, null where the plan lies outside the calibration (plan says why); the order, tolerance and tail they belong to are in result.hartree_boundary. after_scf is the estimate from the final density: status (estimated, not converged or no fit), the shell sums, energy_ry, decay_per_bohr, fit_rms and rough, and the radius and tolerance that would meet the wall share of domain_energy_tolerance_ry. The backend details hold the same under domain_*. per_rank[].domain_rule_seconds is what the default rule took in the parser of a rank, the loading of the pseudopotentials included; that of the root is inside reported_program_seconds, and all are zero for an input with a radius.',
                'With PARSEC_HARTREE_BOUNDARY_CHECK the check runs after the last SCF step and before run_scf returns. Its seconds are hartree_boundary_check_seconds: outside scf_seconds, inside post_scf_finalization_reporting_seconds and reported_program_seconds, and a phase of their own in the stack. Its device arrays are inside the GPU memory peak, so a run that is measured for time or memory leaves the check off.',
                'MPI uses NumPy host fields only; MPICH_GPU_SUPPORT_ENABLED=0. CUDA context initialization is inside preparation and reported program time.',
                'Where cuda_context_initialization_overlapped is true (PARSEC_OVERLAP_CUDA_CONTEXTS, the default) a thread created the contexts beside the static preparation: cuda_context_initialization_seconds_in_preparation is the time of that thread, not a stage of preparation_seconds, and once the contexts existed it started the GPU memory sampler and took the first memory observation, during the preparation.',
                'cuda_context_driver_devices lists the devices whose context that thread created through the CUDA driver library, outside the interpreter lock (PARSEC_CUDA_CONTEXT_CREATION=driver, the default), and cuda_context_driver_seconds is its time inside those calls, a part of the thread time. A device is missing where the library could not be used or numbers the devices otherwise: CuPy then created its context with the lock held, as with PARSEC_CUDA_CONTEXT_CREATION=runtime and as without the overlap, where the list is empty.',
                'With that overlap the GPU ionic sums of the root (PARSEC_IONIC_BACKEND=cupy, on their own thread by default) can reach a device before the context thread does. The context is then created inside ionic_setup_seconds, and initial_free_bytes and the first memory observation of that device can include the arrays of the sums. PARSEC_OVERLAP_CUDA_CONTEXTS=0 gives the figures of the contexts alone.',
                'MPI initialization/imports and provenance hashing precede reported time; external module wall is also recorded and does not include Python interpreter/package-import time before this module.',
                'All rank readiness, preparation completion and post-worker joins are inside reported program wall time.',
                'Stack uses only root sequential SCF wall timers and their remainder; concurrent GPU event totals must not be stacked.',
                'Backend eigensolver counters are rank-local sector statistics; root counters alone do not represent all GPUs, and their sum is not critical-path wall time.',
                'memory_observations are endpoint samples before/after preparation, at root iteration callbacks and after workers stop.',
                'command_seconds is the wall time of each MPI command on a rank, from its receipt through its collective calls, waits for other ranks included. density_pool_release counts, for the density builder of the rank, the pool releases after a density, their full and young collections (PARSEC_CUPY_DENSITY_COLLECTION) and the unreachable objects those found, with the seconds of the collections, of the release of the pools and of the whole sector-wise density of the rank (sector_density_seconds, both included). release names how the pools of several devices are emptied (PARSEC_CUPY_DENSITY_RELEASE): by the threads of the devices, side by side, or serial, one after another in the thread that built the density. release_seconds are those of that thread, its wait for the device threads included, which an MPI rank spends behind the collective calls of its density command; device_release_seconds are the sum of what the device threads took, 0 where none released. shared says whether the pools of the devices of a shared sector basis are emptied as well (PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE): keep, the default, or release; shared_kept counts the densities after which they were left, and calls only those after which a pool was emptied, 0 for a rank whose sectors are all shared. sector_pool_release counts, per device, the releases of the unused pool blocks of a finished sector (PARSEC_CUPY_SECTOR_POOL_RELEASE) and the bytes that went back to the driver; it is empty where no device of the rank solves several sectors in turn, each on a stream of its own. Both follow PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES.',
                'orbital_sector_state_storage and orbital_memory_allocator of a rank are what its own solver decided at its first eigensolve (PARSEC_CUPY_SECTOR_STATE_STORAGE, PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR); result.backend holds those of the root alone. sector_state_fit_bytes gives, per device, the bytes that auto of either held against PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION (0.978) of the memory of the device: the vectors of the sectors whose basis lies on it with one buffer of the Ritz step, or the blocks of a shared basis. For a basis on one device it is a lower limit of what the device must hold, without operators, graph buffers, small matrices and the CUDA context. For a shared basis it is what the rule that shares it counts, with a workspace for slabs as wide as their budget allows (two slabs of at most PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES for a sector on two devices): a device can hold less outside its context (366 MiB less were sampled for 10,456 electrons on 16 GPUs). It is null where no auto asked, and where auto decided by the former rule that PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION under environment names: the vectors of the sectors of a device against that fraction of its memory. A serial control names 0.5 where its launcher names none, and PARSEC_CUPY_SPILL_DOWNLOAD_ORDER=C beside it: a basis that it spills to the host is downloaded by the former call, in C order, and that of a run in the order it has.',
                'distributed_state lists the sectors whose basis the devices of the rank share. slab_columns is the widest slab of H X that the last Ritz step of a sector cut: in the step with one tall block and no PARSEC_CUPY_STREAMING_RITZ_BYTES set, at most PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES (2 GiB; 4294967296 gives the former slabs) on two devices and an even share of 8 GiB on four or more. slab_cut says how that step cut them. With one tall block: multiple where every device gave the same slab to a whole round, so that the round is a multiple of PARSEC_CUPY_RITZ_GRAM_MULTIPLE columns (64) and what the blocks hold beyond those rounds went first, or equal where no such slab exists and under a multiple of 1, which gives the former rounds beside the former slabs. With two: equal or full (PARSEC_CUPY_DISTRIBUTED_STATE_SLABS), a slab of that many columns and more cut down to a whole multiple either way; null with three, which cut none. projection_columns is the widest right-hand side of a product of the projection, a round with one tall block and a slab with two. overlap_seconds is the part of seconds.gram that the overlap took, whose slabs the same multiple cuts: what is left of gram is the projection (0 before the first step and with three tall blocks, which form both matrices in one call). A sector on one device has no entry here: the multiple cuts the slabs of its two Gram matrices, inside subspace_ritz_projection_seconds of the backend statistics. to_rows_copies is the order in which the devices issued the copies between them on the last way to their rows: pairs (four devices take a chunk that all of them send in three turns of two pairs, PARSEC_CUPY_EXCHANGE_PAIRS), together, or queued on the one stream of each (PARSEC_CUPY_EXCHANGE_CONCURRENT=0). It names the order that the switches chose, not what every chunk did, and is null before the first.',
                'gpu_memory holds per-rank peaks from continuous nvidia-smi sampling (default every 200 ms) from CUDA initialization to the final join; it includes CUDA contexts and CuPy pool caches. A peak shorter than the interval can be missed.',
                'Process RSS high water uses OS ru_maxrss; default CuPy pool observations exclude alternate/async allocators. Device-used observations include CUDA/library allocations.',
                'Memory callback overhead is inside SCF wall time. CPU/thread resources per rank are recorded; total host CPU allocation can increase with node count.',
                'Preparation max-rank is descriptive, not added to independently maximized stages or root total.',
            ])
        _atomic_json(timing_path, payload)
        print('FULL_SCF_CONVERGED' if result.converged else 'FULL_SCF_NOT_CONVERGED', timing_path, flush=True)
    return comm.bcast(exit_code, root=0)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--devices', type=_parse_devices, default=(0, 1, 2, 3))
    parser.add_argument('--serial-control', action='store_true', help='one-rank unchanged production solver, without MPI sector context')
    parser.add_argument('--backend', choices=('auto', 'cupy'), default='auto')
    parser.add_argument('--symmetry', choices=('auto', 'on', 'off'))
    parser.add_argument('--pp-dir', type=Path)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument('--symmetry-cache', type=Path, metavar='DIRECTORY',
                       help='read and write the symmetry cache in DIRECTORY; it pays only for repeated calculations of '
                            'the same structure and grid, and every rank then loads or builds all sectors')
    cache.add_argument('--no-symmetry-cache', action='store_true', help='use no symmetry cache (the default)')
    parser.add_argument('--quiet', action='store_true')
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    # Read the launcher's mask before any OpenMP-linked import can narrow it.
    initial_affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    initialization_started = time.perf_counter()
    # There are no GPU buffers in the MPI sector protocol. Disabling the
    # GPU transport layer avoids an unused CUDA-runtime dependency and lets
    # CUDA initialize inside the same measured setup region as the CLI.
    os.environ['MPICH_GPU_SUPPORT_ENABLED'] = '0'
    os.environ['PARSEC_CUPY_DEVICES'] = ','.join(map(str, args.devices))
    if args.serial_control:
        _keep_former_routes(os.environ)
    import cupy as cp
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
    if MPI.Is_initialized():
        raise RuntimeError('MPI was initialized before the runner configured its host-field transport')
    provided = MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    comm = MPI.COMM_WORLD
    try:
        if provided < MPI.THREAD_FUNNELED:
            raise RuntimeError('MPI does not provide the required FUNNELED thread support')
        return execute(args, cp, comm, MPI, time.perf_counter()-initialization_started, initial_affinity)
    except BaseException as error:
        print(f'FULL_SCF_FAILED rank={comm.rank}: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        traceback.print_exc()
        comm.Abort(1)
        raise
    finally:
        if not MPI.Is_finalized():
            MPI.Finalize()


if __name__ == '__main__':
    raise SystemExit(main())
