#!/usr/bin/env python3
"""Replay a captured Hamiltonian on distributed row domains, not a full SCF.

First generate a bounded-memory independent CPU reference with --reference.
Then use --run under srun to audit and time the experimental MPI operator.
The capture is immutable: operator.npz, basis.npy and capture.json. Large
orbitals are never gathered in the timed region or in MPI validation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def imports():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import numpy as np
    import scipy.sparse as sp
    from parsec_python.acceleration.experimental.domain_partition import make_partition
    from parsec_python.acceleration.experimental.mpi_domain import DistributedHamiltonian, generalized_ritz
    return np, sp, make_partition, DistributedHamiltonian, generalized_ritz


def load_capture(folder):
    np, sp, *_ = imports()
    folder = Path(folder)
    with np.load(folder / "operator.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    basis = np.load(folder / "basis.npy", mmap_mode="r", allow_pickle=False)
    n = arrays["neighbors"].shape[1]
    if basis.ndim != 2 or basis.shape[0] != n or basis.dtype != np.float64:
        raise ValueError("basis must be an FP64 (operator rows, columns) array")
    metadata = SimpleNamespace(shape=(n, n), neighbors=arrays["neighbors"],
                               coefficient_codes=arrays["codes"],
                               coefficient_palette=arrays["palette"])
    projectors = sp.csr_matrix((arrays["B_data"], arrays["B_indices"], arrays["B_indptr"]),
                               shape=tuple(arrays["B_shape"]))
    capture = json.loads((folder / "capture.json").read_text(encoding="utf-8"))
    if capture.get("complete") is not True:
        raise ValueError("capture.json must mark complete=true")
    return arrays, basis, metadata, projectors, capture


def capture_hashes(folder):
    return {name: digest(Path(folder) / name)
            for name in ("operator.npz", "basis.npy", "capture.json")}


def column_window(args, basis):
    width = max(max(args.widths), args.ritz_width)
    if args.column_start < 0 or args.column_start + width > basis.shape[1]:
        raise ValueError(f"capture has {basis.shape[1]} columns; requested {args.column_start}:{args.column_start + width}")
    return args.column_start, width


def independent_action(arrays, projectors, basis, output, row_chunk, column_chunk):
    """Canonical stencil order + V + B diag(signs) B.T; no MPI prototype calls."""
    import numpy as np
    n, width = basis.shape
    for first in range(0, width, column_chunk):
        last = min(width, first + column_chunk)
        x = np.asarray(basis[:, first:last])
        coefficients = (projectors.T @ x) * arrays["signs"][:, None]
        for row in range(0, n, row_chunk):
            stop = min(n, row + row_chunk)
            result = np.zeros((stop - row, last - first), dtype=np.float64)
            for neighbors, codes in zip(arrays["neighbors"][:, row:stop], arrays["codes"][:, row:stop]):
                valid = neighbors >= 0
                result[valid] += arrays["palette"][codes[valid], None] * x[neighbors[valid]]
            result += arrays["local_potential"][row:stop, None] * x[row:stop]
            if projectors.shape[1]:
                result += projectors[row:stop] @ coefficients
            output[row:stop, first:last] = result


def serial_ritz(basis, action, width, row_chunk):
    import numpy as np
    import scipy.linalg as la
    overlap = np.zeros((width, width))
    projected = np.zeros_like(overlap)
    for first in range(0, basis.shape[0], row_chunk):
        x = np.asarray(basis[first:first + row_chunk, :width])
        hx = np.asarray(action[first:first + row_chunk, :width])
        overlap += x.T @ x
        projected += x.T @ hx
    overlap = (overlap + overlap.T) * 0.5
    projected = (projected + projected.T) * 0.5
    values, coefficients = la.eigh(projected, overlap, check_finite=True)
    squares = np.zeros(width)
    for first in range(0, basis.shape[0], row_chunk):
        x = np.asarray(basis[first:first + row_chunk, :width])
        hx = np.asarray(action[first:first + row_chunk, :width])
        residual = hx @ coefficients - (x @ coefficients) * values[None, :]
        squares += np.sum(residual * residual, axis=0)
    return dict(eigenvalues=values, residual_norms=np.sqrt(squares), overlap=overlap,
                projected_hamiltonian=projected,
                condition_number=np.array(np.linalg.cond(overlap)))


def make_reference(args):
    np, *_ = imports()
    started = time.perf_counter()
    arrays, basis, metadata, projectors, capture = load_capture(args.capture)
    first, width = column_window(args, basis)
    folder = Path(args.reference_dir or Path(args.capture) / "reference")
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "reference.json").exists() and not args.overwrite_reference:
        raise FileExistsError("reference exists; choose another folder or explicitly use --overwrite-reference")
    hashes = capture_hashes(args.capture)
    load_seconds = time.perf_counter() - started
    action_path = folder / "HX.npy"
    action = np.lib.format.open_memmap(action_path, mode="w+", dtype=np.float64,
                                      shape=(metadata.shape[0], width), fortran_order=True)
    selected = basis[:, first:first + width]
    begun = time.perf_counter()
    independent_action(arrays, projectors, selected, action, args.row_chunk, args.column_chunk)
    action.flush()
    apply_seconds = time.perf_counter() - begun
    begun = time.perf_counter()
    ritz = serial_ritz(selected, action, args.ritz_width, args.row_chunk)
    np.savez(folder / "ritz.npz", **ritz)
    ritz_seconds = time.perf_counter() - begun
    result = dict(schema_version=1, kind="independent_serial_cpu_operator_reference",
                  capture_sha256=hashes, capture=capture, rows=metadata.shape[0],
                  column_start=first, action_columns=width, ritz_columns=args.ritz_width,
                  action_sha256=digest(action_path), ritz_sha256=digest(folder / "ritz.npz"),
                  reference_implementation_sha256=digest(__file__),
                  numpy=np.__version__, load_and_hash_seconds=load_seconds,
                  action_seconds=apply_seconds, ritz_seconds=ritz_seconds,
                  notes="Unchanged captured H; stencil + local V + sparse nonlocal B diag(signs) B.T. No full-SCF claim.")
    atomic_json(folder / "reference.json", result)
    print(f"MPI_DOMAIN_REFERENCE_SAVED {folder}", flush=True)
    return result


def compare_arrays(actual, reference, atol, rtol):
    import numpy as np
    difference = np.abs(np.asarray(actual) - np.asarray(reference))
    allowance = atol + rtol * np.abs(reference)
    finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)))
    return dict(passed=finite and bool(np.all(difference <= allowance)),
                max_absolute_error=float(difference.max(initial=0)),
                max_tolerance_fraction=float((difference / allowance).max(initial=0)),
                absolute_tolerance=atol, relative_tolerance=rtol)


def run_mpi(args):
    # Reuse only the standard-library helper module before MPI initialization.
    helper_path = Path(__file__).with_name("mpi_communication.py")
    spec = importlib.util.spec_from_file_location("_parsec_mpi_audit_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    import cupy as cp
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
    device = helper.visible_device_index(local_rank, cp.cuda.runtime.getDeviceCount())
    cp.cuda.Device(device).use()
    cp.cuda.runtime.free(0)
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
    if MPI.Is_initialized():
        raise RuntimeError("MPI was initialized before GPU selection")
    provided = MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    try:
        np, _, partitioner, Hamiltonian, ritz_solver = imports()
        from parsec_python.acceleration.experimental.host_gather import gather_rows_columns
        operator_type = Hamiltonian
        implementation_options = {}
        if args.implementation in ("overlap", "compact") or args.collective_checks == "fast":
            if args.implementation == "compact":
                from parsec_python.acceleration.experimental.mpi_compact_projectors import CompactProjectorHamiltonian
                operator_type = CompactProjectorHamiltonian
            else:
                from parsec_python.acceleration.experimental.mpi_domain_overlap import OverlappedDistributedHamiltonian
                operator_type = OverlappedDistributedHamiltonian
            implementation_options = dict(
                overlap=args.implementation in ("overlap", "compact"),
                collective_checks=args.collective_checks == "safe",
                validated_widths=tuple(sorted(set(args.widths + [args.ritz_width]))),
            )
        if args.reductions == "nccl":
            from parsec_python.acceleration.experimental.nccl_reductions import NcclReductionMixin
            operator_type = type("Nccl" + operator_type.__name__, (NcclReductionMixin, operator_type), {})
        library = MPI.Get_library_version().strip()
        if not args.allow_non_cray and "cray" not in library.lower():
            raise RuntimeError("Cray MPICH required unless --allow-non-cray is explicit")
        if "cray" in library.lower() and os.environ.get("MPICH_GPU_SUPPORT_ENABLED") != "1":
            raise RuntimeError("GPU-aware replay requires MPICH_GPU_SUPPORT_ENABLED=1")
        if provided < MPI.THREAD_FUNNELED:
            raise RuntimeError("MPI_THREAD_FUNNELED was not provided")
        devices = comm.allgather((socket.gethostname(), cp.cuda.Device(device).pci_bus_id))
        if len(set(devices)) != len(devices):
            raise RuntimeError("two ranks selected the same physical GPU")
        started = time.perf_counter()
        arrays, basis, metadata, projectors, capture = load_capture(args.capture)
        first, width = column_window(args, basis)
        folder = Path(args.reference_dir or Path(args.capture) / "reference")
        reference = json.loads((folder / "reference.json").read_text(encoding="utf-8"))
        if (reference["column_start"] != first or reference["action_columns"] < width
                or reference["ritz_columns"] != args.ritz_width):
            raise ValueError("reference column window does not match requested replay")
        verified = None
        if rank == 0:
            verified = (capture_hashes(args.capture) == reference["capture_sha256"]
                        and digest(folder / "HX.npy") == reference["action_sha256"]
                        and digest(folder / "ritz.npz") == reference["ritz_sha256"])
        if not comm.bcast(verified, root=0):
            raise ValueError("capture/reference SHA256 mismatch")
        expected_hx = np.load(folder / "HX.npy", mmap_mode="r", allow_pickle=False)
        with np.load(folder / "ritz.npz", allow_pickle=False) as data:
            expected_ritz = {key: data[key] for key in data.files}
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        plan = None
        if rank == 0:
            if args.owner_file:
                from parsec_python.acceleration.experimental.domain_partition import PartitionResult, partition_metrics
                external_owner = np.load(args.owner_file, allow_pickle=False)
                external_rows = tuple(np.flatnonzero(external_owner == peer) for peer in range(comm.size))
                halos, metrics = partition_metrics(arrays["neighbors"], external_owner, comm.size,
                                                   local_rows=external_rows)
                metrics.update(method=args.partition, local_order="original_global_row",
                               owner_file_sha256=digest(args.owner_file))
                plan = PartitionResult(external_owner, external_rows, halos, metrics)
            else:
                plan = partitioner(arrays["coordinates"], arrays["neighbors"], comm.size,
                                   args.partition, args.tile_size)
        owner = np.empty(metadata.shape[0], dtype=np.int32)
        if rank == 0:
            owner[:] = plan.owner
        comm.Bcast(owner, root=0)
        rows = np.flatnonzero(owner == rank)
        partition_seconds = time.perf_counter() - started
        measurements = []
        memory = dict(cupy_pool_reserved_high_water_bytes=0,
                      device_memory_used_observed_max_bytes=0)

        def observe_memory():
            cp.cuda.get_current_stream().synchronize()
            memory["cupy_pool_reserved_high_water_bytes"] = max(
                memory["cupy_pool_reserved_high_water_bytes"], cp.get_default_memory_pool().total_bytes())
            available, total = cp.cuda.runtime.memGetInfo()
            memory["device_memory_used_observed_max_bytes"] = max(
                memory["device_memory_used_observed_max_bytes"], total - available)

        def globally_checked(check, label):
            checks = comm.allgather(check)
            if not all(item["passed"] for item in checks):
                raise AssertionError(f"{label} failed on one or more ranks: {checks}")
            return dict(passed=True, max_absolute_error=max(item["max_absolute_error"] for item in checks),
                        max_tolerance_fraction=max(item["max_tolerance_fraction"] for item in checks),
                        absolute_tolerance=check["absolute_tolerance"], relative_tolerance=check["relative_tolerance"])

        def timed(operator, function):
            for _ in range(args.warmups):
                warm = function()
                observe_memory()
                del warm
            before = dict(operator.stats)
            samples = []
            for _ in range(args.repeats):
                comm.Barrier()
                beginning = MPI.Wtime()
                for _ in range(args.iterations):
                    value = function()
                    cp.cuda.get_current_stream().synchronize()
                    del value
                samples.append((MPI.Wtime() - beginning) / args.iterations)
                observe_memory()
            static_keys = {"interior_rows", "boundary_rows"}
            stats = {key: operator.stats[key] - before[key] for key in before if key not in static_keys}
            geometry = {key: operator.stats[key] for key in static_keys if key in operator.stats}
            all_samples = comm.allgather(samples)
            critical = [max(values[index] for values in all_samples) for index in range(args.repeats)]
            return dict(seconds_per_call_max_rank=helper.summarize(critical),
                        per_rank_seconds=all_samples, per_rank_stats=comm.allgather(stats),
                        per_rank_operator_geometry=comm.allgather(geometry),
                        per_rank_projector_partition=comm.allgather(getattr(operator, "projector_partition", {})),
                        per_rank_reduction_metadata=comm.allgather(getattr(operator, "reduction_metadata", {"backend": "mpi"})),
                        timed_call_count=args.repeats * args.iterations)

        transports = ("cuda", "host") if args.transport == "both" else (args.transport,)
        for transport in transports:
            memory = dict(cupy_pool_reserved_high_water_bytes=0,
                          device_memory_used_observed_max_bytes=0)
            begun = time.perf_counter()
            operator = operator_type(metadata, arrays["local_potential"], projectors, arrays["signs"], owner,
                                     comm=comm, xp=cp, transport=transport, local_rows=rows,
                                     **implementation_options)
            setup_seconds = time.perf_counter() - begun
            observe_memory()
            for columns in args.widths:
                begun = time.perf_counter()
                local_basis = cp.asarray(gather_rows_columns(basis, rows, first, first + columns), order="F")
                expected = gather_rows_columns(expected_hx, rows, 0, columns)
                observe_memory()
                input_seconds = time.perf_counter() - begun
                action = operator.apply(local_basis)
                check = globally_checked(compare_arrays(cp.asnumpy(action), expected, args.h_atol, args.h_rtol), "H action")
                observe_memory()
                del action
                measurement = timed(operator, lambda: operator.apply(local_basis))
                measurement.update(kind="hamiltonian_apply", transport=transport, columns=columns,
                                   implementation_class=operator_type.__name__,
                                   accuracy=check, per_rank_input_seconds=comm.allgather(input_seconds))
                measurements.append(measurement)
                del local_basis, expected
            local_basis = cp.asarray(gather_rows_columns(basis, rows, first, first + args.ritz_width), order="F")
            solution = ritz_solver(operator, local_basis)
            eig_check = globally_checked(compare_arrays(solution.eigenvalues, expected_ritz["eigenvalues"], args.eig_atol, 1e-11), "Ritz eigenvalues")
            residual_check = globally_checked(compare_arrays(solution.residual_norms, expected_ritz["residual_norms"], args.residual_atol, args.residual_rtol), "Ritz residual norms")
            # Directly check rotated vectors, not only coefficient-space orthogonality.
            gram_local = cp.asnumpy(solution.local_vectors.T @ solution.local_vectors)
            gram = np.empty_like(gram_local)
            comm.Allreduce(gram_local, gram)
            orthogonality = float(np.max(np.abs(gram - np.eye(args.ritz_width))))
            if not np.isfinite(orthogonality) or orthogonality > args.orth_atol:
                raise AssertionError(f"physical-vector orthogonality {orthogonality} exceeds tolerance")
            projected_check = globally_checked(compare_arrays(solution.projected_hamiltonian,
                expected_ritz["projected_hamiltonian"], args.h_atol, args.h_rtol), "projected H")
            observe_memory()
            del solution
            measurement = timed(operator, lambda: ritz_solver(operator, local_basis))
            measurement.update(kind="generalized_ritz_with_residuals", transport=transport, columns=args.ritz_width,
                               implementation_class=operator_type.__name__,
                               accuracy=dict(eigenvalues=eig_check, residual_norms=residual_check,
                                             projected_hamiltonian=projected_check,
                                             vector_orthogonality_max_absolute=orthogonality,
                                             vector_orthogonality_tolerance=args.orth_atol),
                               per_rank_operator_setup_seconds=comm.allgather(setup_seconds))
            measurements.append(measurement)
            memory_by_rank = comm.allgather(memory)
            for record in measurements:
                if record["transport"] == transport:
                    record["transport_session_memory_by_rank"] = memory_by_rank
            operator.close()
            del local_basis, operator
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        rank_metadata = dict(rank=rank, hostname=socket.gethostname(), gpu_pci_bus_id=cp.cuda.Device(device).pci_bus_id,
                             gpu_name=str(cp.cuda.runtime.getDeviceProperties(device)["name"]),
                             gpu_memory_bytes=int(cp.cuda.runtime.getDeviceProperties(device)["totalGlobalMem"]),
                             local_rows=len(rows), mpi_library=library, cupy=cp.__version__, numpy=np.__version__,
                             cpu_affinity=sorted(os.sched_getaffinity(0)), nics=helper.nic_inventory(),
                             load_and_hash_seconds=load_seconds, partition_seconds=partition_seconds,
                             environment={key: os.environ.get(key) for key in (
                                 "SLURM_JOB_ID", "SLURM_LOCALID", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                 "MPICH_GPU_SUPPORT_ENABLED", "MPICH_OFI_NIC_POLICY", "MPICH_ASYNC_PROGRESS")})
        all_metadata = comm.gather(rank_metadata, root=0)
        if rank == 0:
            source_root = Path(__file__).resolve().parents[1]
            source_paths = [Path(__file__), source_root / "experimental/mpi_domain.py",
                            source_root / "experimental/host_gather.py",
                            source_root / "experimental/domain_partition.py"]
            if operator_type is not Hamiltonian:
                source_paths.append(source_root / "experimental/mpi_domain_overlap.py")
            if args.implementation == "compact":
                source_paths.append(source_root / "experimental/mpi_compact_projectors.py")
            if args.reductions == "nccl":
                source_paths.append(source_root / "experimental/nccl_reductions.py")
            result = dict(schema_version=1, kind="captured_operator_mpi_replay_not_full_scf", status="passed",
                          rank_count=comm.size, node_count=len(set(host for host, _ in devices)),
                          parameters=vars(args), capture=capture, reference=reference,
                          reference_manifest_sha256=digest(folder / "reference.json"),
                          source_sha256={str(path.name): digest(path) for path in source_paths},
                          partition=plan.metrics, metadata=all_metadata, measurements=measurements,
                          notes=["Unchanged captured operator; actual saved basis columns; no timed orbital gather.",
                                 "MPI host transport is the prototype's host staging, not necessarily pinned memory.",
                                 "Includes residual calculation in Ritz timing; collective guard policy is explicit in parameters.",
                                 "base/safe uses the original operator; base/fast uses the new operator with overlap disabled to exercise its reduced collective checks.",
                                 "compact keeps overlap enabled and reduces only projector coefficients with support shared across ranks; projector_partition contains static counts, not byte counters.",
                                 "Use synchronized outer call timings for comparisons; internal apply_seconds can exclude asynchronous GPU completion in fast mode.",
                                 "Reference hashes, loading and ownership setup are outside steady-state timings.",
                                 "Memory is transport-session CuPy reserved-pool high-water and synchronized device observations, not continuously sampled allocation peak.",
                                 "Replay correctness and scaling do not establish full-SCF convergence or speedup."])
            atomic_json(args.output, result)
            print(f"MPI_DOMAIN_REPLAY_PASSED ranks={comm.size} output={args.output}", flush=True)
        comm.Barrier()
    except BaseException as exc:
        print(f"MPI_DOMAIN_REPLAY_FAILED rank={rank}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        comm.Abort(1)
        raise
    finally:
        if not MPI.Is_finalized():
            MPI.Finalize()


def self_test():
    import tempfile
    np, sp, _, Hamiltonian, ritz_solver = imports()
    test_root = Path.cwd().resolve()
    with tempfile.TemporaryDirectory(prefix="parsec-mpi-replay-", dir=test_root) as temporary:
        folder = Path(temporary)
        if not folder.resolve().is_relative_to(test_root):
            raise RuntimeError("self-test temporary directory escaped its working directory")
        n = 23
        neighbors = np.stack((np.arange(n), np.arange(n) - 1, np.arange(n) + 1)).astype(np.int32)
        neighbors[-1, -1] = -1
        codes = np.stack((np.zeros(n), np.ones(n), np.ones(n))).astype(np.uint8)
        projectors = sp.csr_matrix(np.random.default_rng(31).normal(size=(n, 3)) * 0.03)
        np.savez(folder / "operator.npz", coordinates=np.column_stack((np.arange(n), np.zeros(n), np.zeros(n))).astype(np.int32),
                 neighbors=neighbors, codes=codes, palette=np.array([2.0, -1.0]),
                 local_potential=np.arange(n) * 0.01, B_data=projectors.data,
                 B_indices=projectors.indices, B_indptr=projectors.indptr, B_shape=projectors.shape,
                 signs=np.array([1.0, -1.0, 1.0]))
        basis, _ = np.linalg.qr(np.random.default_rng(32).normal(size=(n, 8)))
        np.save(folder / "basis.npy", np.asfortranarray(basis))
        atomic_json(folder / "capture.json", dict(schema=1, complete=True, converged_scf=False, units="Ry"))
        args = SimpleNamespace(capture=str(folder), reference_dir=None, column_start=0,
                               widths=[2, 4, 6], ritz_width=4, row_chunk=7, column_chunk=3,
                               overwrite_reference=False)
        make_reference(args)
        arrays, saved, metadata, projectors, _ = load_capture(folder)
        operator = Hamiltonian(metadata, arrays["local_potential"], projectors, arrays["signs"],
                               np.zeros(n, dtype=np.int32))
        expected = np.load(folder / "reference/HX.npy")
        assert compare_arrays(operator.apply(saved[:, :6]), expected, 1e-12, 1e-12)["passed"]
        solution = ritz_solver(operator, saved[:, :4])
        with np.load(folder / "reference/ritz.npz") as reference:
            assert compare_arrays(solution.eigenvalues, reference["eigenvalues"], 1e-12, 1e-12)["passed"]
            assert compare_arrays(solution.residual_norms, reference["residual_norms"], 1e-12, 1e-12)["passed"]
        assert not compare_arrays(np.array([1.1]), np.array([1.0]), 1e-10, 1e-11)["passed"]
        saved._mmap.close()
    print("MPI_DOMAIN_REPLAY_CPU_SELF_TEST_PASSED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reference", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--capture")
    parser.add_argument("--reference-dir")
    parser.add_argument("--overwrite-reference", action="store_true")
    parser.add_argument("--output", default="mpi_domain_replay.json")
    parser.add_argument("--widths", type=int, nargs="+", default=[6, 24, 96])
    parser.add_argument("--ritz-width", type=int, default=48)
    parser.add_argument("--column-start", type=int, default=0)
    parser.add_argument("--partition", choices=["axis", "brick", "morton", "hilbert", "metis", "metis-node", "brick-node"], default="brick")
    parser.add_argument("--owner-file", help="Offline ownership .npy, validated against this exact row graph")
    parser.add_argument("--tile-size", type=int, default=2)
    parser.add_argument("--transport", choices=["host", "cuda", "both"], default="both")
    parser.add_argument("--reductions", choices=["mpi", "nccl"], default="mpi")
    parser.add_argument("--implementation", choices=["base", "overlap", "compact"], default="base",
                        help="base/safe preserves the original operator; compact reduces only shared projector coefficients with overlap enabled")
    parser.add_argument("--collective-checks", choices=["safe", "fast"], default="safe",
                        help="fast prevalidates widths and requires identical collective call order on all ranks")
    parser.add_argument("--row-chunk", type=int, default=32768)
    parser.add_argument("--column-chunk", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--h-atol", type=float, default=1e-10)
    parser.add_argument("--h-rtol", type=float, default=1e-11)
    parser.add_argument("--eig-atol", type=float, default=1e-8)
    parser.add_argument("--orth-atol", type=float, default=5e-10)
    parser.add_argument("--residual-atol", type=float, default=1e-8)
    parser.add_argument("--residual-rtol", type=float, default=1e-8)
    parser.add_argument("--allow-non-cray", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.capture:
        parser.error("--capture is required")
    if (args.partition.startswith("metis") or args.partition == "brick-node") and not args.owner_file:
        parser.error("Offline partition replay requires an explicit --owner-file")
    if args.reductions == "nccl" and (args.transport != "cuda" or args.collective_checks != "fast"):
        parser.error("experimental NCCL reductions require --transport cuda --collective-checks fast")
    if (min(args.widths) < 1 or not 1 <= args.ritz_width <= 256 or args.row_chunk < 1
            or args.column_chunk < 1 or args.warmups < 0 or args.repeats < 1 or args.iterations < 1):
        parser.error("positive sizes/iterations required; Ritz width must be 1..256")
    if min(args.h_atol, args.h_rtol, args.eig_atol, args.orth_atol, args.residual_atol, args.residual_rtol) <= 0:
        parser.error("accuracy tolerances must be positive")
    (make_reference if args.reference else run_mpi)(args)


if __name__ == "__main__":
    main()
