import argparse
from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from old_architecture.Eigensolvers.pcg import pcg as pcg_cpu
from old_architecture.Eigensolvers.pcg_gpu import pcg as pcg_gpu


def build_problem(n, seed, dtype=np.float32):
    """
    Build the same SPD linear system for both CPU and GPU runs.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, n), dtype=dtype)
    A = base.T @ base
    A += n * np.eye(n, dtype=dtype)
    rhs = rng.standard_normal(n, dtype=dtype)
    x0 = np.zeros(n, dtype=dtype)
    return A, rhs, x0


def relative_residual(A, x, rhs):
    x = np.asarray(x).reshape(-1)
    rhs = np.asarray(rhs).reshape(-1)
    residual = A @ x - rhs
    denom = max(np.linalg.norm(rhs), 1e-12)
    return float(np.linalg.norm(residual) / denom)


def main():
    parser = argparse.ArgumentParser(
        description="Compare CPU and GPU PCG behavior on the same SPD system."
    )
    parser.add_argument("--n", type=int, default=20000, help="Matrix dimension.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--max-iters", type=int, default=100, help="Maximum PCG iterations.")
    parser.add_argument("--tol", type=float, default=1e-6, help="PCG tolerance.")
    args = parser.parse_args()

    print("Building shared test problem...")
    A_cpu, rhs_cpu, x0_cpu = build_problem(args.n, args.seed)

    print(f"Problem size: n={args.n}")
    print(f"Seed: {args.seed}")
    print(f"Tolerance: {args.tol}")
    print(f"Max iterations: {args.max_iters}")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print("")

    cpu_start = time.perf_counter()
    x_cpu, its_cpu = pcg_cpu(A_cpu, rhs_cpu, x0_cpu, m=args.max_iters, tol=args.tol)
    cpu_end = time.perf_counter()
    cpu_time = cpu_end - cpu_start
    cpu_residual = relative_residual(A_cpu, x_cpu, rhs_cpu)

    transfer_start = time.perf_counter()
    A_gpu = cp.asarray(A_cpu)
    rhs_gpu = cp.asarray(rhs_cpu)
    x0_gpu = cp.asarray(x0_cpu)
    cp.cuda.runtime.deviceSynchronize()
    transfer_end = time.perf_counter()

    gpu_start = time.perf_counter()
    x_gpu, its_gpu = pcg_gpu(A_gpu, rhs_gpu, x0_gpu, m=args.max_iters, tol=args.tol)
    cp.cuda.runtime.deviceSynchronize()
    gpu_end = time.perf_counter()

    download_start = time.perf_counter()
    x_gpu_cpu = cp.asnumpy(x_gpu).reshape(-1)
    cp.cuda.runtime.deviceSynchronize()
    download_end = time.perf_counter()

    gpu_transfer_time = transfer_end - transfer_start
    gpu_solve_time = gpu_end - gpu_start
    gpu_download_time = download_end - download_start
    gpu_total_time = gpu_transfer_time + gpu_solve_time + gpu_download_time
    gpu_residual = relative_residual(A_cpu, x_gpu_cpu, rhs_cpu)

    x_cpu = np.asarray(x_cpu).reshape(-1)
    diff = x_cpu - x_gpu_cpu
    diff_norm = float(np.linalg.norm(diff))
    rel_diff = diff_norm / max(float(np.linalg.norm(x_cpu)), 1e-12)
    max_abs_diff = float(np.max(np.abs(diff)))

    print("CPU")
    print(f"  runtime:         {cpu_time:.4f} s")
    print(f"  iterations:      {its_cpu}")
    print(f"  relative resid.: {cpu_residual:.3e}")
    print("")

    print("GPU")
    print(f"  upload time:     {gpu_transfer_time:.4f} s")
    print(f"  solve time:      {gpu_solve_time:.4f} s")
    print(f"  download time:   {gpu_download_time:.4f} s")
    print(f"  total time:      {gpu_total_time:.4f} s")
    print(f"  iterations:      {its_gpu}")
    print(f"  relative resid.: {gpu_residual:.3e}")
    print("")

    print("Comparison")
    print(f"  solve speedup:   {cpu_time / max(gpu_solve_time, 1e-12):.2f}x")
    print(f"  total speedup:   {cpu_time / max(gpu_total_time, 1e-12):.2f}x")
    print(f"  rel. soln diff:  {rel_diff:.3e}")
    print(f"  max abs diff:    {max_abs_diff:.3e}")


if __name__ == "__main__":
    main()
