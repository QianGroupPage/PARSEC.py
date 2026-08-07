import time
from pathlib import Path
import sys

import numpy as np
import cupy as cp

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from old_architecture.Eigensolvers.FermiDirac_gpu import FermiDirac as fd_gpu

# --- parameters ---
n = 5_000_000
EF = 0.5
Temp = 300
Nelec = 1e6

# --- arrays ---
lam_gpu = cp.random.randn(n, dtype=cp.float64)

print(f"Running GPU Fermi-Dirac benchmark... (n={n})")

# --- GPU timing ---
cp.cuda.runtime.deviceSynchronize()
start = time.perf_counter()
fe_gpu, occ_gpu = fd_gpu(lam_gpu, EF, Temp, Nelec)
cp.cuda.runtime.deviceSynchronize()
end = time.perf_counter()

print(f"GPU time: {end - start:.4f}s")
print(f"Deviation (fe): {float(fe_gpu):.4e}")
print(f"Occupancy mean: {float(cp.mean(occ_gpu)):.4f}, std: {float(cp.std(occ_gpu)):.4f}")
