# Benzene comparison

Both calculations use the same geometry, grid, order-12 finite differences,
CA/PZ LDA functional, 20 states, order-one double grid, and canonical PARSEC
C/H pseudopotentials.

## Final energies

| Quantity [Ry] | Python | PARSEC | Python - PARSEC |
|---|---:|---:|---:|
| Eigenvalue energy | -26.80545765 | -26.80659505 | 0.00113740 |
| Hartree energy | 264.06521550 | 264.06624712 | -0.00103162 |
| Integral of `Vxc*rho` | -32.39956806 | -32.39957539 | 0.00000733 |
| Exchange-correlation energy | -24.78505222 | -24.78505779 | 0.00000557 |
| Electron-ion energy | -586.97204043 | -586.97255581 | 0.00051538 |
| Ion-ion energy | 207.86811275 | 207.86811275 | 0.00000000 |
| **Total energy** | **-75.38863311** | **-75.38863314** | **0.00000003** |

The individual SCF-dependent terms differ more than the final total because
each calculation stops on its potential-residual criterion at a different
point in its mixing trajectory. Their double-counting corrections cancel
closely in the total energy.

## Eigenvalues

| Quantity | Python | PARSEC | Python - PARSEC |
|---|---:|---:|---:|
| HOMO, state 15 [Ry] | -0.4821747243 | -0.4821855251 | 0.0000108008 |
| LUMO, state 16 [Ry] | -0.0981525613 | -0.0981788182 | 0.0000262569 |
| HOMO-LUMO gap [Ry] | 0.3840221630 | 0.3840067069 | 0.0000154561 |

Across all 20 states, the maximum absolute eigenvalue difference is
`0.0011148667 Ry` and the RMS difference is `0.0003386347 Ry`. The larger
maximum occurs away from the frontier and includes full-grid splitting of
states that PARSEC treats in separate symmetry representations.

## Convergence and timing

| Quantity | Python | PARSEC |
|---|---:|---:|
| SCF iterations | 15 | 17 |
| Final plain SRE | 0.0062885447 | 0.0026821908 |
| Final charge-weighted SRE | 0.0000947232 | 0.0001465128 |
| Diagonalization time | 53.70 s | 4.89 s |
| Hartree time | 334.81 s | 5.29 s |
| SCF time | 410.91 s | 9.92 s |
| Program wall time | 411.84 s | 10.63 s |

The timing comparison is not like-for-like. Python retains the complete
268,096-point grid and was launched as one Python process. PARSEC used eight
MPI ranks, separated the eight `D_2h` representations, and worked with a
33,512-point reduced domain. PARSEC also reports symmetry labels, forces,
dipoles, and MPI statistics that the current Python single-point path does not
calculate.

The Python result in this table uses the native fixed-cycle CHEBFF
initialization and one source-compatible SUBSPACE filter per later SCF step.
The older adapted ARPACK-first trajectory is no longer part of the package.
