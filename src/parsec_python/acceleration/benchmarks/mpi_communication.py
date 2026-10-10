#!/usr/bin/env python3
"""Audit FP64 GPU MPI transport before enabling distributed SCF algorithms.

Run through scripts/nersc_multinode_run.sh inside an existing allocation.
This is a communication benchmark, not a DFT scaling result. Staged timings
include GPU->pinned-host and pinned-host->GPU copies. Scopes are timed in
sequence; independent groups within each scope execute concurrently.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import sys
import time


def parse_bytes(value):
    text = str(value).strip().lower()
    for suffix, factor in (("gib", 1024**3), ("mib", 1024**2),
                           ("kib", 1024), ("gb", 10**9),
                           ("mb", 10**6), ("kb", 1000), ("b", 1)):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    else:
        factor = 1
    try:
        result = int(text) * factor
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use an integer byte size, e.g. 8KiB") from exc
    if result <= 0 or result % 8:
        raise argparse.ArgumentTypeError("FP64 payload must be a positive multiple of 8 bytes")
    return result


def visible_device_index(local_rank, count):
    """Select the visible ordinal, never reinterpret physical/UUID IDs."""
    if count == 1:
        return 0
    if 0 <= local_rank < count:
        return local_rank
    raise RuntimeError(f"local rank {local_rank} cannot select from {count} visible GPUs")


def summarize(values):
    return {"min": min(values), "median": statistics.median(values),
            "max": max(values), "samples": values}


def self_test():
    assert parse_bytes("8KiB") == 8192
    assert parse_bytes("32MiB") == 33554432
    assert parse_bytes("8") == 8
    for invalid in ("0", "9", "-8", "1.5MiB", "garbage"):
        try:
            parse_bytes(invalid)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(invalid)
    assert visible_device_index(3, 1) == 0
    assert visible_device_index(3, 4) == 3
    try:
        visible_device_index(4, 4)
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid device mapping accepted")
    assert summarize([1.0, 3.0, 2.0])["median"] == 2.0
    print("MPI_COMMUNICATION_CPU_SELF_TEST_PASSED")


def nic_inventory():
    result = []
    for directory in (Path("/sys/class/cxi"), Path("/sys/class/net")):
        if not directory.is_dir():
            continue
        for device in sorted(directory.iterdir()):
            if not (device / "device").exists():
                continue
            item = {"class": directory.name, "name": device.name,
                    "pci_path": str((device / "device").resolve())}
            try:
                item["numa_node"] = (device / "device/numa_node").read_text().strip()
            except OSError:
                pass
            result.append(item)
    return result


def run(args):
    # Importing mpi4py.MPI normally initializes MPI. Select the GPU first so
    # MPICH_OFI_NIC_POLICY=GPU observes the device actually used by this rank.
    import cupy as cp
    import numpy as np
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
    device_index = visible_device_index(local_rank, cp.cuda.runtime.getDeviceCount())
    cp.cuda.Device(device_index).use()
    cp.cuda.runtime.free(0)  # Initialize this CUDA context before MPI_Init.
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
    if MPI.Is_initialized():
        raise RuntimeError("MPI was initialized before GPU selection")
    thread_level = MPI.Init_thread(required=MPI.THREAD_FUNNELED)
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    try:
        mpi_library = MPI.Get_library_version().strip()
        if not args.allow_non_cray and "cray" not in mpi_library.lower():
            raise RuntimeError("expected Cray MPICH; use --allow-non-cray only for other systems")
        if "cray" in mpi_library.lower() and os.environ.get("MPICH_GPU_SUPPORT_ENABLED") != "1":
            raise RuntimeError("set MPICH_GPU_SUPPORT_ENABLED=1 before auditing GPU buffers")
        if thread_level < MPI.THREAD_FUNNELED:
            raise RuntimeError("MPI did not provide MPI_THREAD_FUNNELED")
        stream = cp.cuda.get_current_stream()
        props = cp.cuda.runtime.getDeviceProperties(device_index)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        environment_keys = (
            "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_PROCID", "SLURM_LOCALID",
            "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES", "PARSEC_MPI_ORIGINAL_CUDA_VISIBLE_DEVICES",
            "MPICH_GPU_SUPPORT_ENABLED", "MPICH_OFI_NIC_POLICY", "MPICH_OFI_NIC_VERBOSE",
            "MPICH_ASYNC_PROGRESS", "MPICH_GPU_IPC_ENABLED", "MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED",
            "OMP_NUM_THREADS", "OMP_PLACES", "OMP_PROC_BIND", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        )
        metadata = {
            "rank": rank, "local_rank": local_rank, "hostname": socket.gethostname(),
            "visible_device_index": device_index, "gpu_name": name,
            "gpu_pci_bus_id": cp.cuda.Device(device_index).pci_bus_id,
            "gpu_total_memory_bytes": int(props["totalGlobalMem"]),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "python": sys.version, "platform": platform.platform(), "cupy": cp.__version__,
            "numpy": np.__version__, "mpi4py": mpi4py.__version__, "mpi_library": mpi_library,
            "mpi_thread_level": thread_level, "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver_version": cp.cuda.runtime.driverGetVersion(), "nics": nic_inventory(),
            "environment": {key: os.environ.get(key) for key in environment_keys},
        }
        all_metadata = world.allgather(metadata)
        physical_devices = [(item["hostname"], item["gpu_pci_bus_id"]) for item in all_metadata]
        if len(set(physical_devices)) != len(physical_devices):
            raise RuntimeError("multiple MPI ranks mapped to the same physical GPU")
        node = world.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
        if len(set(world.allgather(node.Get_size()))) != 1:
            raise RuntimeError("benchmark requires the same number of ranks on every node")
        node_leader = node.bcast(rank if node.Get_rank() == 0 else None, root=0)
        leaders = sorted(set(world.allgather(node_leader)))
        node_index = leaders.index(node_leader)
        # These groups exercise all node NICs concurrently with one GPU/rank.
        lane = world.Split(color=node.Get_rank(), key=node_index)
        scopes = [("world", world), ("within_node", node), ("across_nodes_lane", lane)]
        groups = []
        skipped = []
        for scope_index, (scope, comm) in enumerate(scopes):
            world.Barrier()
            size = comm.Get_size()
            members = comm.allgather(rank)
            if size < 2:
                if comm.Get_rank() == 0:
                    skipped.append({"scope": scope, "members": members, "reason": "fewer than two ranks"})
                continue
            local_records = []
            for nbytes in args.sizes:
                count = nbytes // 8
                value = float(comm.Get_rank() + 1)
                g_send = cp.full(count, value, dtype=cp.float64)
                g_recv = cp.empty_like(g_send)
                # Keep the pinned allocations alive throughout the measurements.
                p_send = cp.cuda.alloc_pinned_memory(nbytes)
                p_recv = cp.cuda.alloc_pinned_memory(nbytes)
                h_send = np.frombuffer(p_send, dtype=np.float64, count=count)
                h_recv = np.frombuffer(p_recv, dtype=np.float64, count=count)
                stream.synchronize()
                for op_index, operation in enumerate(("allreduce", "neighbor_sendrecv")):
                    tag = 310 + scope_index * 10 + op_index
                    expected = (float(size * (size + 1) // 2) if operation == "allreduce"
                                else float((comm.Get_rank() - 1) % size + 1))

                    def communicate(send, recv):
                        if operation == "allreduce":
                            comm.Allreduce([send, MPI.DOUBLE], [recv, MPI.DOUBLE], op=MPI.SUM)
                        else:
                            comm.Sendrecv([send, MPI.DOUBLE], dest=(comm.Get_rank() + 1) % size,
                                          sendtag=tag, recvbuf=[recv, MPI.DOUBLE],
                                          source=(comm.Get_rank() - 1) % size, recvtag=tag)

                    for transport in ("gpu_aware", "pinned_host_staging"):
                        def step():
                            stream.synchronize()
                            if transport == "gpu_aware":
                                communicate(g_send, g_recv)
                            else:
                                g_send.get(out=h_send, stream=stream, blocking=True)
                                communicate(h_send, h_recv)
                                g_recv.set(h_recv, stream=stream)
                            stream.synchronize()

                        # Poison the receive buffer and validate before timing.
                        # These integer-valued FP64 sums must be exactly equal.
                        g_recv.fill(float("nan"))
                        step()
                        correct = bool(cp.all(g_recv == expected).item())
                        if not comm.allreduce(correct, op=MPI.LAND):
                            raise AssertionError(f"{scope}/{nbytes}/{operation}/{transport} failed validation")
                        for _ in range(args.warmups):
                            step()
                        local_seconds = []
                        for _ in range(args.repeats):
                            # Keep concurrent node/lane groups on the same
                            # benchmark case, rather than letting faster groups
                            # interfere with a different case on slower groups.
                            world.Barrier()
                            started = MPI.Wtime()
                            for _ in range(args.iterations):
                                step()
                            local_seconds.append((MPI.Wtime() - started) / args.iterations)
                        world.Barrier()
                        # Validate again after timing, not in the timed region.
                        correct = bool(cp.all(g_recv == expected).item())
                        if not comm.allreduce(correct, op=MPI.LAND):
                            raise AssertionError("post-timing validation failed")
                        rank_seconds = comm.gather(local_seconds, root=0)
                        if comm.Get_rank() == 0:
                            critical_seconds = [max(values[i] for values in rank_seconds)
                                                for i in range(args.repeats)]
                            local_records.append({
                                "operation": operation, "transport": transport,
                                "payload_bytes": nbytes, "dtype": "float64", "validated_exact": True,
                                "seconds_per_operation_max_rank": summarize(critical_seconds),
                                "rank_seconds_per_operation": dict(zip(map(str, members), rank_seconds)),
                                "payload_gbytes_per_second": nbytes / statistics.median(critical_seconds) / 1e9,
                            })
                del g_send, g_recv, h_send, h_recv, p_send, p_recv
            if comm.Get_rank() == 0:
                groups.append({"scope": scope, "members": members, "records": local_records})
            world.Barrier()
        gathered_groups = world.gather(groups, root=0)
        gathered_skipped = world.gather(skipped, root=0)
        if rank == 0:
            flat_groups = [group for per_rank in gathered_groups for group in per_rank]
            aggregate = []
            for scope, _ in scopes:
                scope_groups = [group for group in flat_groups if group["scope"] == scope]
                if not scope_groups:
                    continue
                for index, template in enumerate(scope_groups[0]["records"]):
                    samples = [max(group["records"][index]["seconds_per_operation_max_rank"]["samples"][j]
                                   for group in scope_groups) for j in range(args.repeats)]
                    aggregate.append({"scope": scope, "group_count": len(scope_groups),
                                      "operation": template["operation"], "transport": template["transport"],
                                      "payload_bytes": template["payload_bytes"],
                                      "seconds_per_operation_max_all_ranks": summarize(samples)})
            result = {
                "schema_version": 1, "benchmark": "mpi_gpu_communication", "status": "passed",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rank_count": world.Get_size(), "node_count": len(leaders),
                "parameters": vars(args), "metadata": all_metadata, "groups": flat_groups,
                "aggregate": aggregate,
                "skipped": [item for per_rank in gathered_skipped for item in per_rank],
                "timing_notes": [
                    "Blocking collectives and producer/consumer stream synchronization are included.",
                    "Pinned-host staging includes both GPU/host copies; pinned allocation is excluded.",
                    "Each sample excludes the preceding world barrier and validation.",
                    "Independent per-node or per-lane groups execute concurrently, not in isolation.",
                    "Payload throughput is bytes/time, not normalized collective network bus bandwidth.",
                    "No DFT accuracy or SCF scaling conclusions follow from this transport audit alone.",
                ],
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            temporary.replace(output)
            print(f"MPI_GPU_COMMUNICATION_PASSED ranks={world.Get_size()} nodes={len(leaders)} output={output}")
        lane.Free()
        node.Free()
        world.Barrier()
    except BaseException as exc:
        print(f"MPI_GPU_COMMUNICATION_FAILED rank={rank}: {exc}", file=sys.stderr, flush=True)
        world.Abort(1)
        raise
    finally:
        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.Finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=parse_bytes, nargs="+", default=[8192, 1048576, 8388608, 33554432])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", default="mpi_communication.json")
    parser.add_argument("--allow-non-cray", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="CPU-only argument/mapping self-test; needs no MPI/CUDA")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.warmups < 0 or args.repeats < 1 or args.iterations < 1:
        parser.error("warmups must be nonnegative; repeats and iterations must be positive")
    run(args)


if __name__ == "__main__":
    main()
