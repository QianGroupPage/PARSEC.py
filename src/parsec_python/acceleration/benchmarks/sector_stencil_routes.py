"""Build the packed sector stencils of one input on the host, route by route.

No GPU and no SCF.  For every route of ``--routes`` (``csr``, ``direct``,
``numpy``; see ``PARSEC_SECTOR_STENCIL``) a fresh process prepares what a
sector rank prepares on the host: the grid, the full-grid Laplacian or its
descriptor, the projectors, the symmetry maps and the reduced operators of
``--sectors``.  It prints the stage times, the high-water mark of the process
and a SHA-256 of the neighbors, codes and palette of each sector.  The exit
status is 0 only if every route gave the same arrays.

The arrays of ``direct`` and ``numpy`` are those of ``csr`` with its native
reduction, ``PARSEC_NATIVE_SECTOR_ASSEMBLY=1``, which is what this benchmark
runs when the variable is unset.  With the variable at 0 the SciPy reduction
adds repeated columns of a row in another order and a few entries can differ
in the last bits, so a comparison with ``csr`` is refused.

The ``csr`` route holds the full-grid matrix twice while it is built: about
32 bytes for every stencil entry of the grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _high_water_bytes():
    """Resident high-water mark of this process (peak working set on Windows)."""
    try:
        import resource
    except ImportError:
        return _peak_working_set_bytes()
    scale = 1 if sys.platform == 'darwin' else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


def _peak_working_set_bytes():
    if sys.platform != 'win32':
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)


def _digest(stencil):
    value = hashlib.sha256()
    value.update(repr(tuple(stencil.neighbors.shape)).encode('ascii'))
    for array in (stencil.neighbors, stencil.coefficient_codes, stencil.coefficient_palette):
        value.update(memoryview(array).cast('B'))
    return value.hexdigest()


def measure(source, route, sectors=None, pp_dir=None):
    """Prepare the sector operators of ``source`` on one route in this process."""
    settings = {'PARSEC_SECTOR_STENCIL': route, 'PARSEC_IONIC_BACKEND': 'native',
                'PARSEC_ACCELERATED_RESIDENT': '0'}
    # The former route as every measured run took it.
    settings['PARSEC_NATIVE_SECTOR_ASSEMBLY'] = os.environ.get('PARSEC_NATIVE_SECTOR_ASSEMBLY', '1')
    with patch.dict(os.environ, settings):
        from parsec_python.Input import parse_parsec_input
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.Symmetry import (
            load_or_build_reflection_decomposition, load_or_detect_reflection_reduction)
        from parsec_python.acceleration.Symmetry.operator_cache import load_or_build_reduced_operators
        from parsec_python.acceleration.backends.selection import BackendSelection

        translation = parse_parsec_input(source, pseudopotential_directory=pp_dir)
        if translation.ignore_symmetry:
            raise ValueError('the input disables symmetry; there are no sectors to build')
        selection = BackendSelection(requested='auto', selected='cupy',
                                     finite_difference_builder='native', hartree_backend='native')
        deferral = {} if route == 'csr' else dict(defer_native_laplacian=True,
                                                 deferred_laplacian_cache_directory=None)
        started = time.perf_counter()
        reference = driver._prepare_reference_physics(
            translation.problem, selection, orbital_operators_only=True, **deferral)
        symmetry_started = time.perf_counter()
        reduction, _ = load_or_detect_reflection_reduction(
            reference.grid, reference.atoms, cache_directory=None)
        if reduction.group_order <= 1:
            raise ValueError('no nontrivial symmetry was detected; there are no sectors to build')
        decomposition, _ = load_or_build_reflection_decomposition(
            reference.grid, reduction, reduction_key=None, cache_directory=None)
        symmetry_seconds = time.perf_counter()-symmetry_started
        bundle = load_or_build_reduced_operators(
            decomposition, reference.negative_laplacian, reference.nonlocal_operator,
            cache_directory=None, representations=sectors)
        total = time.perf_counter()-started
        built = [index for index, item in enumerate(bundle.stencil_metadata) if item is not None]
        return dict(
            route=route, stencil_builder=bundle.cache_info.stencil_builder,
            native_sector_assembly=settings['PARSEC_NATIVE_SECTOR_ASSEMBLY'] if route == 'csr' else None,
            grid_points=int(reference.grid.size), laplacian_nnz=int(reference.negative_laplacian.nnz),
            sector_dimensions=list(decomposition.sector_sizes), sectors=built,
            grid_seconds=reference.timings.grid_seconds,
            finite_difference_seconds=reference.timings.finite_difference_seconds,
            projector_seconds=reference.timings.nonlocal_ionic_seconds,
            symmetry_seconds=symmetry_seconds,
            operator_build_seconds=bundle.cache_info.build_seconds,
            host_preparation_seconds=total,
            full_grid_matrix_built=bool(getattr(reference.negative_laplacian, 'materialized', True)),
            process_rss_high_water_bytes=_high_water_bytes(),
            threads=os.environ.get('OMP_NUM_THREADS'),
            sha256={str(index): _digest(bundle.stencil_metadata[index]) for index in built})


def _sectors(text):
    try:
        return tuple(int(item) for item in text.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('sectors must be comma-separated integers') from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--pp-dir', type=Path)
    parser.add_argument('--sectors', type=_sectors, help='default: every sector')
    parser.add_argument('--routes', default='csr,direct', help='comma-separated; default csr,direct')
    parser.add_argument('--route', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.route is not None:
        print(json.dumps(measure(args.input, args.route, args.sectors, args.pp_dir)), flush=True)
        return 0
    routes = [route.strip() for route in args.routes.split(',')]
    if 'csr' in routes and len(routes) > 1 and os.environ.get('PARSEC_NATIVE_SECTOR_ASSEMBLY', '1') != '1':
        print('PARSEC_NATIVE_SECTOR_ASSEMBLY must be 1 or unset to compare csr with the other routes bit for bit',
              file=sys.stderr)
        return 2
    rows = []
    for route in routes:
        command = [sys.executable, str(Path(__file__).resolve()),
                   '--input', str(args.input), '--route', route]
        if args.pp_dir is not None:
            command += ['--pp-dir', str(args.pp_dir)]
        if args.sectors is not None:
            command += ['--sectors', ','.join(map(str, args.sectors))]
        completed = subprocess.run(command, stdout=subprocess.PIPE, text=True)
        if completed.returncode:
            print(f'route {route} failed with status {completed.returncode}', file=sys.stderr)
            return 2
        rows.append(json.loads(completed.stdout.strip().splitlines()[-1]))
        print(json.dumps(rows[-1]), flush=True)
    if len(rows) < 2:
        print('SECTOR_STENCILS_NOT_COMPARED', flush=True)
        return 0
    same = all(row['sha256'] == rows[0]['sha256'] for row in rows)
    print('SECTOR_STENCILS_IDENTICAL' if same else 'SECTOR_STENCILS_DIFFER', flush=True)
    return 0 if same else 1


if __name__ == '__main__':
    raise SystemExit(main())
