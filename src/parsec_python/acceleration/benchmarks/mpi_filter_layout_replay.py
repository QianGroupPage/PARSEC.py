#!/usr/bin/env python3
"""Compare row-domain and orbital-column Chebyshev components, not full SCF.

Run --reference on one GPU, after mpi_domain_replay.py --reference. Then
run --run with one MPI rank per GPU. The capture and both references remain
immutable during MPI execution. No orbital matrix is gathered on any rank.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time


def sibling(name):
    spec = importlib.util.spec_from_file_location("_filter_replay_" + name,
                                                  Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def imports():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import numpy as np
    from parsec_python.acceleration.backends.cupy_stencil_major import StencilMajorHostMetadata
    from parsec_python.acceleration.experimental.domain_partition import make_partition
    from parsec_python.acceleration.experimental.mpi_compact_projectors import CompactProjectorHamiltonian
    from parsec_python.acceleration.experimental.mpi_domain import chebyshev_filter
    from parsec_python.acceleration.experimental.orbital_layout import ColumnLayout, column_chebyshev_filter
    return np, StencilMajorHostMetadata, make_partition, CompactProjectorHamiltonian, chebyshev_filter, ColumnLayout, column_chebyshev_filter


def load_capture(args):
    helper = sibling("mpi_domain_replay")
    arrays, basis, _, projectors, capture = helper.load_capture(args.capture)
    np, Metadata, *_ = imports()
    metadata = Metadata((basis.shape[0], basis.shape[0]), arrays["neighbors"],
                        arrays["codes"], arrays["palette"])
    if args.column_start < 0 or args.width < 1 or args.column_start + args.width > basis.shape[1]:
        raise ValueError("requested column window exceeds the captured basis")
    eigenvalues = np.load(Path(args.capture) / "eigenvalues.npy", allow_pickle=False)
    if eigenvalues.ndim != 1 or not eigenvalues.size or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("capture eigenvalues must be a finite, nonempty vector")
    return helper, arrays, basis, metadata, projectors, capture, eigenvalues


def source_hashes(helper):
    root = Path(__file__).resolve().parents[1]
    relative = ("experimental/orbital_layout.py", "experimental/mpi_domain.py",
                "experimental/host_gather.py",
                "experimental/mpi_domain_overlap.py", "experimental/mpi_compact_projectors.py",
                "experimental/domain_partition.py", "backends/cupy.py",
                "backends/cupy_stencil_major.py", "backends/cupy_projectors.py")
    result = {name: helper.digest(root / name) for name in relative}
    result["benchmarks/mpi_filter_layout_replay.py"] = helper.digest(__file__)
    return result


def capture_hashes(helper, folder):
    result = helper.capture_hashes(folder)
    result["eigenvalues.npy"] = helper.digest(Path(folder) / "eigenvalues.npy")
    return result


def conservative_bounds(arrays, projectors, eigenvalues):
    """Absolute row-sum bound including every nonlocal projector term."""
    import numpy as np
    row_bound = np.abs(arrays["local_potential"]).copy()
    for neighbors, codes in zip(arrays["neighbors"], arrays["codes"]):
        active = neighbors >= 0
        row_bound[active] += np.abs(arrays["palette"][codes[active]])
    absolute_b = abs(projectors)
    if projectors.shape[1]:
        coefficients = np.asarray(absolute_b.sum(axis=0)).ravel() * np.abs(arrays["signs"])
        row_bound += absolute_b @ coefficients
    upper = float(np.max(row_bound))
    upper += 32 * np.finfo(np.float64).eps * max(1.0, abs(upper))
    lower, reference = float(np.max(eigenvalues) + 0.5), float(np.min(eigenvalues))
    if not np.all(np.isfinite([reference, lower, upper])) or not reference < lower < upper:
        raise ValueError(f"invalid filter bounds: reference={reference}, lower={lower}, upper={upper}")
    return dict(lower_bound=lower, upper_bound=upper, reference_eigenvalue=reference)


def explicit_filter(apply, x, degree, bounds):
    """Independent unfused normalized recurrence for reference generation."""
    half = (bounds["upper_bound"] - bounds["lower_bound"]) / 2
    center = (bounds["upper_bound"] + bounds["lower_bound"]) / 2
    sigma_one = half / (bounds["reference_eigenvalue"] - center)
    sigma = sigma_one
    previous = x.copy()
    current = (apply(x) - center * x) * (sigma_one / half)
    for _ in range(2, degree + 1):
        sigma_next = 1 / (2 / sigma_one - sigma)
        following = sigma_next * ((2 / half) * (apply(current) - center * current) - sigma * previous)
        previous, current, sigma = current, following, sigma_next
    return current


def compare_blocks(actual, expected, xp, row_chunk, atol, rtol):
    """Bound host copies; summarize every element, not a sample."""
    import numpy as np
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch {actual.shape} != {expected.shape}")
    finite, passed, maximum, fraction, diff_sq, reference_sq = True, True, 0.0, 0.0, 0.0, 0.0
    for first in range(0, actual.shape[0], row_chunk):
        block = actual[first:first + row_chunk]
        host = np.asarray(block) if xp is np else xp.asnumpy(block)
        reference = np.asarray(expected[first:first + row_chunk])
        difference = np.abs(host - reference)
        allowance = atol + rtol * np.abs(reference)
        finite = finite and bool(np.all(np.isfinite(host)) and np.all(np.isfinite(reference)))
        passed = passed and bool(np.all(difference <= allowance))
        maximum = max(maximum, float(np.max(difference, initial=0)))
        fraction = max(fraction, float(np.max(difference / allowance, initial=0)))
        diff_sq += float(np.sum(difference * difference))
        reference_sq += float(np.sum(reference * reference))
    return dict(passed=finite and passed, finite=finite, max_absolute_error=maximum,
                max_tolerance_fraction=fraction, squared_error=diff_sq,
                squared_reference_norm=reference_sq, absolute_tolerance=atol, relative_tolerance=rtol)


def combine_checks(checks):
    import math
    error = sum(item["squared_error"] for item in checks)
    reference = sum(item["squared_reference_norm"] for item in checks)
    return dict(passed=all(item["passed"] for item in checks),
                max_absolute_error=max(item["max_absolute_error"] for item in checks),
                max_tolerance_fraction=max(item["max_tolerance_fraction"] for item in checks),
                error_frobenius_norm=math.sqrt(error), reference_frobenius_norm=math.sqrt(reference),
                relative_frobenius_error=math.sqrt(error / reference) if reference else None,
                absolute_tolerance=checks[0]["absolute_tolerance"], relative_tolerance=checks[0]["relative_tolerance"])


def select_gpu():
    helper = sibling("mpi_communication")
    import cupy as cp
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
    device = helper.visible_device_index(local_rank, cp.cuda.runtime.getDeviceCount())
    cp.cuda.Device(device).use()
    cp.cuda.runtime.free(0)
    return cp, device, helper


def make_reference(args):
    if int(os.environ.get("SLURM_NTASKS", "1")) != 1:
        raise RuntimeError("--reference requires a one-task launch, never an MPI multi-task step")
    cp, device, _ = select_gpu()
    started = time.perf_counter()
    helper, arrays, basis, metadata, projectors, capture, eigenvalues = load_capture(args)
    np, *_ = imports()
    from parsec_python.acceleration.backends.cupy import CuPyHamiltonian
    folder = Path(args.reference_dir or Path(args.capture) / "filter_reference")
    if any((folder / name).exists() for name in ("filter_reference.json", "filtered.npy")) and not args.overwrite_reference:
        raise FileExistsError("filter reference exists; use a new folder or --overwrite-reference")
    hashes = capture_hashes(helper, args.capture)
    cpu_folder = Path(args.cpu_reference_dir or Path(args.capture) / "reference")
    cpu = json.loads((cpu_folder / "reference.json").read_text(encoding="utf-8"))
    old_hashes = {name: hashes[name] for name in ("operator.npz", "basis.npy", "capture.json")}
    if cpu["capture_sha256"] != old_hashes or helper.digest(cpu_folder / "HX.npy") != cpu["action_sha256"]:
        raise ValueError("independent CPU H reference does not match the capture")
    offset = args.column_start - cpu["column_start"]
    if offset < 0 or offset + args.width > cpu["action_columns"]:
        raise ValueError("independent CPU H reference does not cover the requested columns")
    expected_h = np.load(cpu_folder / "HX.npy", mmap_mode="r", allow_pickle=False)
    bounds = conservative_bounds(arrays, projectors, eigenvalues)
    operator = CuPyHamiltonian(None, arrays["local_potential"], (projectors, arrays["signs"]),
                              retain_generic_laplacian=False, finite_difference_metadata=metadata)
    x = cp.asarray(basis[:, args.column_start:args.column_start + args.width], order="F")
    hx = operator.apply(x)
    cp.cuda.get_current_stream().synchronize()
    h_check = combine_checks([compare_blocks(hx, expected_h[:, offset:offset + args.width], cp,
                                             args.row_chunk, 1e-10, 1e-11)])
    if not h_check["passed"]:
        raise AssertionError(f"production GPU H does not match CPU reference: {h_check}")
    del hx
    setup_seconds = time.perf_counter() - started
    beginning = time.perf_counter()
    filtered = explicit_filter(operator.apply, x, args.degree, bounds)
    cp.cuda.get_current_stream().synchronize()
    filter_seconds = time.perf_counter() - beginning
    if not bool(cp.all(cp.isfinite(filtered))):
        raise FloatingPointError("reference filter produced nonfinite values")
    folder.mkdir(parents=True, exist_ok=True)
    # Withdraw completion before replacing either file in explicit overwrite mode.
    manifest_path = folder / "filter_reference.json"
    if manifest_path.exists():
        manifest_path.unlink()
    output = np.lib.format.open_memmap(folder / "filtered.npy", mode="w+", dtype=np.float64,
                                      shape=filtered.shape, fortran_order=True)
    for first in range(0, args.width, args.column_chunk):
        output[:, first:first + args.column_chunk] = cp.asnumpy(filtered[:, first:first + args.column_chunk])
    output.flush()
    result = dict(schema_version=1, kind="serial_production_gpu_chebyshev_reference_not_full_scf", complete=True,
                  capture=capture, capture_sha256=hashes, column_start=args.column_start,
                  width=args.width, degree=args.degree, bounds=bounds,
                  filtered_sha256=helper.digest(folder / "filtered.npy"),
                  cpu_reference_manifest_sha256=helper.digest(cpu_folder / "reference.json"),
                  production_h_vs_independent_cpu=h_check, source_sha256=source_hashes(helper),
                  load_hash_and_setup_seconds=setup_seconds, serial_filter_seconds=filter_seconds,
                  gpu_pci_bus_id=cp.cuda.Device(device).pci_bus_id, cupy=cp.__version__, numpy=np.__version__,
                  notes=["Uniform-degree explicit normalized Chebyshev recurrence using production CuPyHamiltonian.apply.",
                         "Upper bound is max absolute stencil row sum + abs(V) + abs(B) diag(abs(signs)) abs(B).T 1, with roundoff padding.",
                         "The capture uses Euclidean-normalized symmetry-sector states; no multiplicity weighting.",
                         "Filtering a fixed captured Hamiltonian does not establish full-SCF accuracy or acceleration."])
    helper.atomic_json(manifest_path, result)
    print(f"FILTER_LAYOUT_REFERENCE_PASSED output={folder}", flush=True)


def run_mpi(args):
    # The CUDA context must be selected before MPI_Init for Cray GPU/NIC affinity.
    cp, device, timing_helper = select_gpu()
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
    if MPI.Is_initialized():
        raise RuntimeError("MPI was initialized before selecting the GPU")
    provided = MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.rank
    try:
        np, _, partitioner, Compact, row_filter, Columns, column_filter = imports()
        from parsec_python.acceleration.experimental.host_gather import gather_rows_columns
        library = MPI.Get_library_version().strip()
        if provided < MPI.THREAD_FUNNELED:
            raise RuntimeError("MPI_THREAD_FUNNELED unavailable")
        if not args.allow_non_cray and "cray" not in library.lower():
            raise RuntimeError("expected Cray MPICH; override with --allow-non-cray")
        if "cray" in library.lower() and os.environ.get("MPICH_GPU_SUPPORT_ENABLED") != "1":
            raise RuntimeError("Cray GPU replay requires MPICH_GPU_SUPPORT_ENABLED=1")
        devices = comm.allgather((socket.gethostname(), cp.cuda.Device(device).pci_bus_id))
        if len(set(devices)) != len(devices):
            raise RuntimeError("multiple ranks selected the same physical GPU")
        started = time.perf_counter()
        helper, arrays, basis, metadata, projectors, capture, _ = load_capture(args)
        folder = Path(args.reference_dir or Path(args.capture) / "filter_reference")
        reference = json.loads((folder / "filter_reference.json").read_text(encoding="utf-8"))
        if not reference.get("complete") or any(reference[key] != getattr(args, key) for key in ("degree", "width", "column_start")):
            raise ValueError("filter reference parameters do not match the request")
        valid = None
        if rank == 0:
            valid = (capture_hashes(helper, args.capture) == reference["capture_sha256"]
                     and helper.digest(folder / "filtered.npy") == reference["filtered_sha256"])
        if not comm.bcast(valid, root=0):
            raise ValueError("capture/filter reference SHA256 mismatch")
        expected = np.load(folder / "filtered.npy", mmap_mode="r", allow_pickle=False)
        if expected.shape != (basis.shape[0], args.width):
            raise ValueError("invalid filter reference array shape")
        bounds = reference["bounds"]
        if not bounds["reference_eigenvalue"] < bounds["lower_bound"] < bounds["upper_bound"]:
            raise ValueError("invalid reference bounds")
        load_seconds = time.perf_counter() - started
        parameters = dict(degree=args.degree, **bounds)
        measurements = []
        memory = {}

        def observe_memory():
            cp.cuda.get_current_stream().synchronize()
            available, total = cp.cuda.runtime.memGetInfo()
            for key, value in (("cupy_pool_reserved_high_water_bytes", cp.get_default_memory_pool().total_bytes()),
                               ("device_memory_used_observed_max_bytes", total - available)):
                memory[key] = max(memory.get(key, 0), int(value))

        def check(actual, target, label):
            local = compare_blocks(actual, target, cp, args.row_chunk, args.atol, args.rtol)
            result = combine_checks(comm.allgather(local))
            if not result["passed"]:
                raise AssertionError(f"{label} did not match serial filter: {result}")
            return result

        def timed(operator, function, kind, accuracy):
            for _ in range(args.warmups):
                value = function()
                observe_memory()
                del value
            before = dict(operator.stats)
            samples = []
            for _ in range(args.repeats):
                cp.cuda.get_current_stream().synchronize()
                comm.Barrier()
                beginning = MPI.Wtime()
                for _ in range(args.iterations):
                    value = function()
                    cp.cuda.get_current_stream().synchronize()
                    del value
                samples.append((MPI.Wtime() - beginning) / args.iterations)
                observe_memory()
            all_samples = comm.allgather(samples)
            critical = [max(values[index] for values in all_samples) for index in range(args.repeats)]
            counters = {key: value - before[key] for key, value in operator.stats.items()
                        if key not in {"interior_rows", "boundary_rows"}}
            result = dict(kind=kind, columns=args.width, degree=args.degree, transport=args.transport,
                          seconds_per_call_max_rank=timing_helper.summarize(critical),
                          per_rank_seconds=all_samples, per_rank_stats=comm.allgather(counters),
                          timed_call_count=args.repeats * args.iterations, accuracy=accuracy)
            measurements.append(result)
            comm.Barrier()

        begun = time.perf_counter()
        plan = partitioner(arrays["coordinates"], arrays["neighbors"], comm.size, "brick") if rank == 0 else None
        owner = np.empty(basis.shape[0], dtype=np.int32)
        if rank == 0:
            owner[:] = plan.owner
        comm.Bcast(owner, root=0)
        rows = np.flatnonzero(owner == rank)
        row_operator = Compact(metadata, arrays["local_potential"], projectors, arrays["signs"], owner,
                               comm=comm, xp=cp, transport=args.transport, local_rows=rows,
                               overlap=True, collective_checks=False, validated_widths=(args.width,))
        row_basis = cp.asarray(gather_rows_columns(basis, rows, args.column_start,
                                                 args.column_start + args.width), order="F")
        row_setup_seconds = time.perf_counter() - begun
        observe_memory()
        result = row_filter(row_operator, row_basis, **parameters)
        row_check = check(result, gather_rows_columns(expected, rows), "brick row-domain filter")
        observe_memory()
        del result
        timed(row_operator, lambda: row_filter(row_operator, row_basis, **parameters),
              "brick_compact_overlap_fast_filter", row_check)
        row_memory = comm.allgather(dict(memory))
        projector_partition = comm.allgather(row_operator.projector_partition)
        row_operator.close()
        del row_operator, row_basis
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        comm.Barrier()

        memory = {}
        begun = time.perf_counter()
        columns = Columns(metadata, arrays["local_potential"], projectors, arrays["signs"], args.width,
                          comm=comm, xp=cp, transport=args.transport)
        start, stop = columns.column_start, columns.column_stop
        column_basis = cp.asarray(basis[:, args.column_start + start:args.column_start + stop], order="F")
        column_setup_seconds = time.perf_counter() - begun
        column_result = column_filter(columns, column_basis, **parameters)
        column_check = check(column_result, expected[:, start:stop], "column filter")
        row_result = columns.columns_to_rows(column_result)
        row_layout_check = check(row_result, expected[columns.row_start:columns.row_stop], "columns-to-rows")
        returned = columns.rows_to_columns(row_result)
        roundtrip_check = check(returned, expected[:, start:stop], "rows-to-columns roundtrip")
        observe_memory()
        del returned

        def roundtrip():
            temporary = columns.columns_to_rows(column_result)
            return columns.rows_to_columns(temporary)

        def combined():
            filtered = column_filter(columns, column_basis, **parameters)
            temporary = columns.columns_to_rows(filtered)
            return columns.rows_to_columns(temporary)

        combined_result = combined()
        combined_check = check(combined_result, expected[:, start:stop], "column filter plus redistribution roundtrip")
        del combined_result
        timed(columns, lambda: column_filter(columns, column_basis, **parameters), "column_filter", column_check)
        timed(columns, lambda: columns.columns_to_rows(column_result), "columns_to_contiguous_rows", row_layout_check)
        timed(columns, lambda: columns.rows_to_columns(row_result), "contiguous_rows_to_columns", roundtrip_check)
        timed(columns, roundtrip, "column_row_column_roundtrip", roundtrip_check)
        timed(columns, combined, "column_filter_plus_roundtrip", combined_check)
        column_memory = comm.allgather(dict(memory))
        column_counts = columns.column_counts.tolist()
        contiguous_row_counts = columns.row_counts.tolist()
        columns.close()
        rank_metadata = dict(rank=rank, hostname=socket.gethostname(), gpu_pci_bus_id=cp.cuda.Device(device).pci_bus_id,
                             gpu_name=str(cp.cuda.runtime.getDeviceProperties(device)["name"]),
                             gpu_memory_bytes=int(cp.cuda.runtime.getDeviceProperties(device)["totalGlobalMem"]),
                             brick_rows=len(rows), orbital_columns=stop - start, mpi_library=library,
                             cupy=cp.__version__, numpy=np.__version__, cpu_affinity=sorted(os.sched_getaffinity(0)),
                             load_and_hash_seconds=load_seconds, row_setup_seconds=row_setup_seconds,
                             column_setup_seconds=column_setup_seconds,
                             environment={key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_LOCALID",
                                          "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MPICH_GPU_SUPPORT_ENABLED",
                                          "MPICH_OFI_NIC_POLICY", "MPICH_ASYNC_PROGRESS")})
        metadata_by_rank = comm.gather(rank_metadata, root=0)
        if rank == 0:
            data = dict(schema_version=1, kind="captured_filter_layout_mpi_replay_not_full_scf", status="passed",
                        rank_count=comm.size, node_count=len(set(host for host, _ in devices)),
                        parameters=vars(args), capture=capture, reference=reference,
                        reference_manifest_sha256=helper.digest(folder / "filter_reference.json"),
                        source_sha256=source_hashes(helper), partition=plan.metrics,
                        projector_partition=projector_partition, metadata=metadata_by_rank,
                        orbital_columns_by_rank=column_counts, contiguous_rows_by_rank=contiguous_row_counts,
                        row_session_memory_by_rank=row_memory, column_session_memory_by_rank=column_memory,
                        measurements=measurements,
                        notes=["Component replay only: no density update, Hartree, Ritz, SCF convergence, or total-energy comparison.",
                               "Both filters use the same captured full Hamiltonian, input columns, uniform degree and spectral bounds.",
                               "Row filter uses brick ownership, compact shared-projector reductions, halo overlap and fast collective checks.",
                               "Column filter replicates Hamiltonian metadata but stores only its own orbital columns. Each H has zero MPI calls; filter parameter validation still performs control collectives.",
                               "Column redistributions use contiguous rows, not brick ownership. Roundtrip includes two full Alltoallv layout changes but no Ritz computation.",
                               "Combined time is measured directly; it is not a sum of independent medians. GPU producers and final consumers are synchronized.",
                               "All owned elements are checked against the serial production-GPU reference before timing. Only scalar diagnostics are gathered.",
                               "Memory observations are separate row/column sessions: retained CuPy pool high-water and synchronized device use, not continuously sampled peaks. Column session also retains filtered inputs for standalone transfer timings.",
                               "MPI host transport is prototype host staging and is not guaranteed pinned.",
                               "Filter tolerance is an arithmetic replay tolerance, not a claim of relaxed physical accuracy."])
            helper.atomic_json(args.output, data)
            print(f"FILTER_LAYOUT_REPLAY_PASSED ranks={comm.size} output={args.output}", flush=True)
        comm.Barrier()
    except BaseException as exc:
        print(f"FILTER_LAYOUT_REPLAY_FAILED rank={rank}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        comm.Abort(1)
        raise
    finally:
        if not MPI.Is_finalized():
            MPI.Finalize()


def self_test(args):
    """Read-only serial CPU integration on a supplied synthetic capture."""
    helper, arrays, basis, metadata, projectors, _, eigenvalues = load_capture(args)
    np, _, _, Compact, row_filter, Columns, column_filter = imports()
    bounds = conservative_bounds(arrays, projectors, eigenvalues)
    x = np.asfortranarray(basis[:, args.column_start:args.column_start + args.width])
    columns = Columns(metadata, arrays["local_potential"], projectors, arrays["signs"], args.width)
    row_operator = Compact(metadata, arrays["local_potential"], projectors, arrays["signs"],
                           np.zeros(basis.shape[0], dtype=np.int32), collective_checks=False,
                           validated_widths=(args.width,))
    expected = explicit_filter(columns.apply, x, args.degree, bounds)
    for actual in (row_filter(row_operator, x, degree=args.degree, **bounds),
                   column_filter(columns, x, degree=args.degree, **bounds),
                   columns.rows_to_columns(columns.columns_to_rows(expected))):
        check = combine_checks([compare_blocks(actual, expected, np, args.row_chunk, 1e-12, 1e-12)])
        if not check["passed"]:
            raise AssertionError(check)
    bad = expected.copy()
    bad[0, 0] += 0.1
    if combine_checks([compare_blocks(bad, expected, np, args.row_chunk, 1e-12, 1e-12)])["passed"]:
        raise AssertionError("comparison failed to reject an injected error")
    row_operator.close()
    columns.close()
    print("FILTER_LAYOUT_REPLAY_CPU_SELF_TEST_PASSED", json.dumps(bounds), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reference", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--capture", required=True)
    parser.add_argument("--reference-dir")
    parser.add_argument("--cpu-reference-dir")
    parser.add_argument("--overwrite-reference", action="store_true")
    parser.add_argument("--output", default="mpi_filter_layout_replay.json")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--column-start", type=int, default=0)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--transport", choices=("cuda", "host"), default="cuda")
    parser.add_argument("--row-chunk", type=int, default=32768)
    parser.add_argument("--column-chunk", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--allow-non-cray", action="store_true")
    args = parser.parse_args()
    import math
    if (min(args.width, args.degree, args.row_chunk, args.column_chunk, args.repeats, args.iterations) < 1
            or args.column_start < 0 or args.warmups < 0):
        parser.error("positive dimensions, degree and repeats required; nonnegative start/warmups")
    if not all(math.isfinite(value) and value > 0 for value in (args.atol, args.rtol)):
        parser.error("finite positive arithmetic tolerances required")
    (self_test if args.self_test else make_reference if args.reference else run_mpi)(args)


if __name__ == "__main__":
    main()
