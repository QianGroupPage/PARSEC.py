# H2 physical comparison result

Date: 2026-07-28

> Historical result: this comparison predates the native strict-CHEBFF
> rewrite. It is retained as evidence for the full radial-potential and
> nonlocal-projector implementation, but its eigensolver iteration count is
> not evidence for the current solver. Use `h2_canonical_nodg` and
> `0d_benzene` for current end-to-end solver validation.

Both calculations used the files in this directory: the same 861-point
Martins-new `H_POTRE.DAT`, H-H distance, real-space grid, finite-difference
order, CA-LDA functional, temperature, state count, and convergence criterion.

The PARSEC values come from `parsec_reference.out`. The Python values come
from `parsec_python.out`. Differences below are Python minus PARSEC.

| Quantity | PARSEC [Ry] | Python [Ry] | Signed difference [Ry] |
|---|---:|---:|---:|
| Total energy | -2.293197280000 | -2.293197288860 | -8.8600e-09 |
| Eigenvalue energy | -1.511660570000 | -1.511614227359 | +4.6343e-05 |
| Hartree energy | 2.589281990000 | 2.589369800355 | +8.7810e-05 |
| Integral Vxc*rho | -1.695393350000 | -1.695446216525 | -5.2867e-05 |
| Exchange-correlation energy | -1.298685970000 | -1.298726189116 | -4.0219e-05 |
| Electron-ion energy | -7.045473040000 | -7.045648842046 | -1.7580e-04 |
| Ion-ion energy | 1.411138670000 | 1.411138666667 | -3.3330e-09 |
| Occupied eigenvalue | -0.755830283600 | -0.755807113700 | +2.3170e-05 |

Convergence and runtime:

| Quantity | PARSEC | Python |
|---|---:|---:|
| Converged SCF iteration | 19 | 7 |
| Final weighted SRE [Ry] | 6.58059e-05 | 5.304882e-05 |
| Full active grid points | 179,944 | 179,944 |
| Grid points diagonalized directly | symmetry sectors of 22,493 | full 179,944 |
| Nonlocal projector columns | 6 | 6 |
| Wall time [s] | 19.30 | 190.12 |

## Interpretation

The total energy is reproduced to about nine decimal places in Rydberg for
this matched physical case. The separate energy terms agree at the
`1e-4 Ry` scale; their leading differences cancel in the stationary total
energy. The occupied eigenvalue differs by `2.32e-5 Ry`, below the requested
`1e-4` diagonalization tolerance.

This archived run used the earlier global full-grid ARPACK initialization and
Chebyshev reuse path, whereas PARSEC diagonalized point-group representations
separately. The current Python implementation no longer uses that solver.
Unoccupied states were not paired one-for-one in this historical table because
PARSEC allocates states among symmetry representations while Python returned
the globally lowest full-grid states. Degenerate subspaces may also rotate.

This result validates a full radial potential and the nonlocal KB path. It
does not validate PARSEC's Ono-Hirose double-grid algorithm.

## Canonical PARSEC H potential

The stock `/home/zeyizhang/PARSEC/tests/H2` case uses PARSEC's canonical
863-point, one-channel H potential and `Double_Grid_Order: 3`. Python currently
rejects that double-grid setting. A trial order-one PARSEC input with that
one-channel potential crashes in PARSEC's `nonloc` routine before SCF, so it
cannot supply a matched order-one reference. The stock canonical result must
remain a future target until the Python double-grid implementation is added.
