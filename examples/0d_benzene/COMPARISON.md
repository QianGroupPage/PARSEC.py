# Benzene comparison

Both calculations use the same geometry, grid, order-12 finite differences,
CA/PZ LDA functional, 20 states, order-one double grid, and canonical PARSEC
C/H pseudopotentials. The Python values come from `parsec_python.out`, run
with the code of 5 November 2026 on a laptop GPU (RTX 5070).

## Final energies

| Quantity [Ry] | Python | PARSEC | Python - PARSEC |
|---|---:|---:|---:|
| Eigenvalue energy | -26.80047913 | -26.80659505 | 0.00611592 |
| Hartree energy | 264.06795951 | 264.06624712 | 0.00171239 |
| Integral of `Vxc*rho` | -32.39989016 | -32.39957539 | -0.00031477 |
| Exchange-correlation energy | -24.78529630 | -24.78505779 | -0.00023851 |
| Electron-ion energy | -586.97566381 | -586.97255581 | -0.00310800 |
| Ion-ion energy | 207.86811275 | 207.86811275 | 0.00000000 |
| **Total energy** | **-75.38863333** | **-75.38863314** | **-0.00000019** |

The individual SCF-dependent terms differ more than the final total because
each calculation stops on its potential-residual criterion at a different
point in its mixing trajectory. Their double-counting corrections cancel
closely in the total energy.

## Eigenvalues

| Quantity | Python | PARSEC | Python - PARSEC |
|---|---:|---:|---:|
| HOMO, state 15 [Ry] | -0.4819995091 | -0.4821855251 | 0.0001860160 |
| LUMO, state 16 [Ry] | -0.0979891855 | -0.0981788182 | 0.0001896327 |
| HOMO-LUMO gap [Ry] | 0.3840103236 | 0.3840067069 | 0.0000036167 |

Across all 20 states, the maximum absolute eigenvalue difference is
`0.0003613903 Ry` (state 18) and the RMS difference is `0.0002051253 Ry`.
The differences are close to a common shift of about `2e-4 Ry` of all
levels, which comes from where each SCF loop stops: with
`Convergence_Criterion: 1.0e-8` the Python HOMO is `-0.4820798948 Ry`, and
the levels of the run at the input's criterion lie 0.8e-4 to 1.1e-4 Ry above
the tightly converged ones for states 1 to 17 and up to 1.8e-4 Ry above for
the three highest.

## Convergence and timing

| Quantity | Python | PARSEC |
|---|---:|---:|
| SCF iterations | 10 | 17 |
| Final plain SRE | 0.0065269999 | 0.0026821908 |
| Final charge-weighted SRE | 0.0001034149 | 0.0001465128 |
| Diagonalization time | 3.61 s | 4.89 s |
| Hartree time | 1.12 s | 5.29 s |
| SCF time | 5.43 s | 9.92 s |
| Program wall time | 15.12 s | 10.63 s |

The timing comparison is not like-for-like. Python ran as one process on a
laptop GPU and solved the eight `D_2h` representations there; its program
time includes about 10 s of CUDA initialization and set-up. PARSEC used eight
MPI ranks on the CPU, separated the same eight representations, and worked
with a 33,512-point reduced domain. PARSEC also reports forces and MPI
statistics that the current Python single-point path does not calculate.

The Python result in this table uses the native CHEBFF initialization
(`FF_MaxIter` 3, the default since 5 November 2026) and one source-compatible
SUBSPACE filter per later SCF step. The older adapted ARPACK-first trajectory
and the full-grid run of 28 July 2026 (15 SCF iterations, `-75.38863311 Ry`)
are no longer part of the package.
