"""Densities from orbital blocks, on any array namespace (NumPy on the host, CuPy on a device).

The k-point drivers form their densities with these.  ``vectors`` may live where the eigensolver
keeps it (a device) or hold only some of the rows of the grid (a row-distributed basis): the
results have the length of the rows given, and a row-distributed caller puts the pieces together.
The sums run over column chunks so that the temporaries stay a few columns wide however large the
basis is.
"""

from __future__ import annotations

import numpy as np

COLUMN_CHUNK = 64


def weighted_square_sum(vectors, occupations, xp=np, chunk: int = COLUMN_CHUNK):
    """``sum_n f_n |v_n(row)|**2`` for every row of ``vectors`` (shape ``(rows, states)``)."""
    occupations = np.asarray(occupations, dtype=np.float64)
    total = xp.zeros(int(vectors.shape[0]), dtype=xp.float64)
    for c0 in range(0, occupations.size, chunk):
        c1 = min(c0 + chunk, occupations.size)
        if not np.any(occupations[c0:c1]):
            continue
        block = vectors[:, c0:c1]
        total += (block.real * block.real + block.imag * block.imag) @ xp.asarray(occupations[c0:c1])
    return total


def spinor_sums(spinors, occupations, xp=np, chunk: int = COLUMN_CHUNK):
    """``(sum f|up|^2, sum f|down|^2, sum f up^* down)`` for stacked spinors ``(2*rows, states)``."""
    occupations = np.asarray(occupations, dtype=np.float64)
    rows = int(spinors.shape[0]) // 2
    up_sum = xp.zeros(rows, dtype=xp.float64)
    down_sum = xp.zeros(rows, dtype=xp.float64)
    cross_sum = xp.zeros(rows, dtype=xp.complex128)
    for c0 in range(0, occupations.size, chunk):
        c1 = min(c0 + chunk, occupations.size)
        if not np.any(occupations[c0:c1]):
            continue
        weights = xp.asarray(occupations[c0:c1])
        up = spinors[:rows, c0:c1]
        down = spinors[rows:, c0:c1]
        up_sum += (up.real * up.real + up.imag * up.imag) @ weights
        down_sum += (down.real * down.real + down.imag * down.imag) @ weights
        cross_sum += (xp.conj(up) * down) @ weights
    return up_sum, down_sum, cross_sum


__all__ = ["COLUMN_CHUNK", "spinor_sums", "weighted_square_sum"]
