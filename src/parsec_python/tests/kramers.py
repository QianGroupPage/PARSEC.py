"""Time reversal at a time-reversal-invariant k-point, checked where it is defined: per degenerate level.

At a TRIM point (Gamma, a cluster) without a magnetic field every level is even-fold degenerate and its
states carry no net spin.  The spin of one state is fixed only inside a level of two states: the sum
over a Kramers pair of a doublet is gauge invariant, but a quartet (the j = 3/2 level of a gold atom in a
cubic cell, for instance) is any unitary mixture of its four states, and the sum over two of them depends
on the basis the eigensolver returned.  Up to 2026-11-05 the tests summed ``<S_z>`` over consecutive
pairs; the quartet at -2.3065 Ry of the Gamma-point gold cell of
``test_periodic_self_consistent_soc_cli`` gave pair sums of +/-0.030 on the CPU, +/-0.019 with the device
solver on the host and +/-0.051 on an RTX 5070 (the bound was 0.05), while the sum over the level was
below 1e-4 on every route.
"""

from __future__ import annotations

import numpy as np

from parsec_python.Output.bands import degenerate_levels


def assert_level_moments_cancel(
    case, eigenvalues, moments, *, checked: int, moment_delta: float = 0.05, tolerance: float = 1.0e-4
) -> None:
    """The moments of every degenerate level (states within ``tolerance`` Ry) among the lowest
    ``checked`` states sum to zero within ``moment_delta``.  A level that reaches past ``checked``
    (it may be cut off at the top of the computed states) is not checked, as the pair loops did not
    check the top pair."""

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    moments = np.asarray(moments, dtype=float)
    for level in degenerate_levels(eigenvalues, tolerance):
        if int(level.max()) >= checked:
            continue
        case.assertAlmostEqual(
            float(np.sum(moments[level])),
            0.0,
            delta=moment_delta,
            msg=f"spin of the level at {eigenvalues[level[0]]:.6f} Ry (states {level.tolist()})",
        )


__all__ = ["assert_level_moments_cancel"]
