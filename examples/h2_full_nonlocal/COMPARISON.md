# H2 physical comparison result

Date: 2026-11-05 (Python run). The comparison of 2026-07-28 used an
earlier Python program, which diagonalized the full grid with an ARPACK first
solve and converged in 7 SCF iterations to -2.293197288860 Ry; that program
is no longer part of the package.

Both calculations used the files in this directory: the same 861-point
Martins-new `H_POTRE.DAT`, H-H distance, real-space grid, finite-difference
order, CA-LDA functional, temperature, state count, and convergence criterion.

The PARSEC values come from `parsec_reference.out`. The Python values come
from `parsec_python.out`: `python src\parsec_python\main.py
examples\h2_full_nonlocal\parsec.in --no-archive --log parsec_python.out`
with the code of 5 November 2026 on a laptop GPU (RTX 5070). The report
prints the energy terms with eight decimals. Differences below are Python
minus PARSEC.

| Quantity | PARSEC [Ry] | Python [Ry] | Signed difference [Ry] |
|---|---:|---:|---:|
| Total energy | -2.2931972800 | -2.2931972700 | +1.0000e-08 |
| Eigenvalue energy | -1.5116605700 | -1.5116575100 | +3.0600e-06 |
| Hartree energy | 2.5892819900 | 2.5894781600 | +1.9617e-04 |
| Integral Vxc*rho | -1.6953933500 | -1.6955114800 | -1.1813e-04 |
| Exchange-correlation energy | -1.2986859700 | -1.2987759800 | -9.0010e-05 |
| Electron-ion energy | -7.0454730400 | -7.0458385800 | -3.6554e-04 |
| Ion-ion energy | 1.4111386700 | 1.4111386700 | +0.0000e+00 |
| Occupied eigenvalue | -0.7558302836 | -0.7558287540 | +1.5296e-06 |

Convergence and runtime:

| Quantity | PARSEC | Python |
|---|---:|---:|
| Converged SCF iteration | 19 | 17 |
| Final weighted SRE [Ry] | 6.58059e-05 | 6.39849e-05 |
| Full active grid points | 179,944 | 179,944 |
| Grid points diagonalized directly | eight symmetry sectors of 22,493 | eight symmetry sectors (D_2h) |
| Nonlocal projector columns | 6 | 6 |
| Wall time [s] | 19.30 (CPU) | 6.42 (laptop GPU) |

## Interpretation

The total energy is reproduced to about eight decimal places in Rydberg for
this matched physical case. The separate energy terms agree at the
`1e-4 Ry` scale; their leading differences cancel in the stationary total
energy, and each code stops on its potential-residual criterion at a
different point of its mixing trajectory. The occupied eigenvalue differs by
`1.5e-6 Ry`, below the requested `1e-4` diagonalization tolerance.

Both codes diagonalize the eight representations of `D_2h` separately; the
Python first solve is CHEBFF. Unoccupied states are not paired one-for-one
in this table because the two codes distribute the 16 states differently
among the representations, and unoccupied states are converged less tightly.
Degenerate subspaces may also rotate.

This result validates a full radial potential and the nonlocal KB path. It
does not validate PARSEC's Ono-Hirose double-grid algorithm.

## Canonical PARSEC H potential

The stock `<PARSEC_FORTRAN_ROOT>/tests/H2` case uses PARSEC's canonical
863-point, one-channel H potential and `Double_Grid_Order: 3`. Python currently
rejects that double-grid setting. A trial order-one PARSEC input with that
one-channel potential crashes in PARSEC's `nonloc` routine before SCF, so it
cannot supply a matched order-one reference. The stock canonical result must
remain a future target until the Python double-grid implementation is added.
