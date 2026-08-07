"""GPU-resident finite-difference Laplacian builder.

This mirrors :mod:`Laplacian.fd3d` for the GPU path, but avoids the large
host-side dense temporaries used by the legacy assembly. The operator is built
as a sparse Kronecker sum of 1D finite-difference second-derivative stencils,
which maps directly onto the x-fastest flattening used elsewhere in the code.
"""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as cpsparse


def _second_derivative_coefficients(degree: int) -> tuple[float, dict[int, float]]:
    """Return the 1D centered finite-difference stencil coefficients."""
    if degree <= 2:
        return 2.0, {1: -1.0}
    if degree <= 4:
        return 5.0 / 2.0, {1: -4.0 / 3.0, 2: 1.0 / 12.0}
    if degree <= 6:
        return 49.0 / 18.0, {1: -3.0 / 2.0, 2: 3.0 / 20.0, 3: -1.0 / 90.0}
    return 205.0 / 72.0, {1: -8.0 / 5.0, 2: 1.0 / 5.0, 3: -8.0 / 315.0, 4: 1.0 / 560.0}


def _fd1d(n: int, degree: int, dtype) -> cpsparse.csr_matrix:
    """Build the 1D second-derivative stencil as a CSR matrix on the GPU."""
    center, neighbors = _second_derivative_coefficients(degree)
    diagonals = [cp.full(n, center, dtype=dtype)]
    offsets = [0]

    for offset, value in sorted(neighbors.items()):
        width = n - offset
        if width <= 0:
            continue
        band = cp.full(width, value, dtype=dtype)
        diagonals.extend([band, band.copy()])
        offsets.extend([offset, -offset])

    return cpsparse.diags(diagonals, offsets, shape=(n, n), format="csr", dtype=dtype)


def fd3d_gpu(nx: int, ny: int, nz: int, degree: int, scale: float = 1.0, dtype=cp.float32):
    """Construct the 3D finite-difference Laplacian directly on the GPU."""
    dx = _fd1d(nx, degree, dtype)
    dy = _fd1d(ny, degree, dtype)
    dz = _fd1d(nz, degree, dtype)

    ix = cpsparse.eye(nx, dtype=dtype, format="csr")
    iy = cpsparse.eye(ny, dtype=dtype, format="csr")
    iz = cpsparse.eye(nz, dtype=dtype, format="csr")

    iy_ix = cpsparse.kron(iy, ix, format="csr")
    x_term = cpsparse.kron(iz, cpsparse.kron(iy, dx, format="csr"), format="csr")
    y_term = cpsparse.kron(iz, cpsparse.kron(dy, ix, format="csr"), format="csr")
    z_term = cpsparse.kron(dz, iy_ix, format="csr")

    laplacian = x_term + y_term + z_term
    if scale != 1.0:
        laplacian = laplacian * dtype(scale)
    return laplacian.tocsr()


def warmup() -> None:
    """Trigger the first-use CuPy JIT work before timed solver sections."""
    trial = fd3d_gpu(2, 2, 2, 2)
    probe = cp.ones(trial.shape[0], dtype=cp.float32)
    _ = trial @ probe
    cp.cuda.Stream.null.synchronize()
