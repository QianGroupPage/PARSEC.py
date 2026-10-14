# 1s binding energies of the 2022 set

Written by `analyze.py --write` from the logs in `cases/`.

## Atomic energies of the pseudopotential generation (Table 1, spin-unpolarized), eV

| element | all-electron atom | paper | all-electron ion | paper | pseudo-atom | paper | pseudo-ion | paper |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | -669.486 | -669.486 | -462.410 | -462.410 | -70.483 | -70.483 | -128.688 | -128.688 |
| C | -1027.632 | -1027.632 | -723.254 | -723.254 | -145.942 | -145.943 | -240.665 | -240.665 |
| N | -1481.742 | -1481.741 | -1061.468 | -1061.468 | -262.446 | -262.446 | -401.742 | -401.742 |
| O | -2040.902 | -2040.902 | -1486.074 | -1486.074 | -428.168 | -428.168 | -619.405 | -619.405 |

## Reference molecules: absolute binding energies, eV

The paper's Table S1 uses atomic energies of spin-polarized atoms. The constant per element that this takes off a binding energy follows from its Table 1 (B 6.436, C 8.558, N 10.083, O 10.468) and is applied in the fifth column.

| set | molecule | method | here | paper Table 2 | here, constant applied | paper Table S1 | difference |
|---|---|---|---:|---:|---:|---:|---:|
| B | P(CH3)3BH3 | PP-PBE | 198.042 | 198.04 | 191.606 | 191.60 | +0.006 |
| B | P(CH3)3BH3 | PP-PBE(B3LYP) | 198.236 |  | 191.800 | 191.77 | +0.030 |
| C | CH3SiH3 | PP-PBE | 296.718 | 296.71 | 288.160 | 288.16 | -0.000 |
| C | CH3SiH3 | PP-PBE(B3LYP) | 296.723 |  | 288.165 | 288.15 | +0.015 |
| N | (C2H5)2NH | PP-PBE | 411.945 | 411.94 | 401.862 | 401.86 | +0.002 |
| N | (C2H5)2NH | PP-PBE(B3LYP) | 411.725 |  | 401.642 | 401.64 | +0.002 |
| O | S(CH3)2O | PP-PBE | 544.931 | 544.93 | 534.463 | 534.46 | +0.003 |
| O | S(CH3)2O | PP-PBE(B3LYP) | 544.563 |  | 534.095 | 534.09 | +0.005 |

## B 1s: shifts relative to P(CH3)3BH3, eV

| molecule | experiment | PP-PBE here | paper | difference | PP-PBE(B3LYP) here | paper | difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| P(CH3)3BH3 | 0.00 | 0.000 | 0.00 | +0.000 | 0.000 | 0.00 | +0.000 |
| H3BN(CH2CH3)3 | 0.27 | 0.240 | 0.24 | +0.000 | 0.300 | 0.29 | +0.010 |
| BH3CO | 2.17 | 2.173 | 2.17 | +0.003 | 2.208 | 2.21 | -0.002 |
| B(CH2CH3)3 | 2.57 | 2.183 | 2.18 | +0.003 | 2.334 | 2.35 | -0.016 |
| B(OC2H5)3 | 4.67 | 4.413 | 4.42 | -0.007 | 4.870 | 4.87 | -0.000 |
| B(OCH3)3 | 4.87 | 4.678 | 4.68 | -0.002 | 5.133 | 5.13 | +0.003 |
| BBr3 | 6.07 | 5.823 | 5.83 | -0.007 | 6.147 | 6.17 | -0.023 |
| BCl3 | 6.87 | 6.448 | 6.45 | -0.002 | 6.867 | 6.88 | -0.013 |
| BF3 | 9.87 | 9.350 | 9.35 | +0.000 | 10.001 | 10.01 | -0.009 |

## C 1s: shifts relative to CH3SiH3, eV

| molecule | experiment | PP-PBE here | paper | difference | PP-PBE(B3LYP) here | paper | difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| CH3SiH3 | 0.00 | 0.000 | 0.00 | +0.000 | 0.000 | 0.00 | +0.000 |
| CH4 | 0.49 | 0.428 | 0.43 | -0.002 | 0.526 | 0.53 | -0.004 |
| CH3SH | 1.10 | 1.099 | 1.09 | +0.009 | 1.145 | 1.14 | +0.005 |
| CH3OH | 2.11 | 2.045 | 2.04 | +0.005 | 2.186 | 2.18 | +0.006 |
| H2NCN | 3.15 | 2.975 | 2.98 | -0.005 | 3.285 | 3.30 | -0.015 |
| HCN | 3.19 | 3.130 | 3.13 | -0.000 | 3.311 | 3.32 | -0.009 |
| NH2CHO | 4.14 | 3.888 | 3.89 | -0.002 | 4.311 | 4.30 | +0.011 |
| HNCO | 5.58 | 5.338 | 5.34 | -0.002 | 5.957 | 5.96 | -0.003 |
| CO | 5.89 | 6.002 | 6.01 | -0.008 | 6.351 | 6.36 | -0.009 |
| CH2F2 | 6.05 | 5.781 | 5.78 | +0.001 | 6.124 | 6.12 | +0.004 |
| CCl4 | 6.08 | 5.874 | 5.89 | -0.016 | 6.013 | 6.05 | -0.037 |
| CO2 | 7.38 | 7.090 | 7.10 | -0.010 | 7.820 | 7.82 | -0.000 |
| CHF3 | 8.79 | 8.435 | 8.44 | -0.005 | 8.961 | 8.96 | +0.001 |
| COF2 | 9.33 | 8.886 | 8.89 | -0.004 | 9.615 | 9.62 | -0.005 |
| CF4 | 11.54 | 10.995 | 11.00 | -0.005 | 11.712 | 11.72 | -0.008 |

## N 1s: shifts relative to (C2H5)2NH, eV

| molecule | experiment | PP-PBE here | paper | difference | PP-PBE(B3LYP) here | paper | difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| (C2H5)2NH | 0.00 | 0.000 | 0.00 | +0.000 | 0.000 | 0.00 | +0.000 |
| H2N(CH2)3OCH3 | 0.30 | 0.344 | 0.35 | -0.006 | 0.278 | 0.28 | -0.002 |
| CH3CH2NH2 | 0.35 | 0.419 | 0.39 | +0.029 | 0.493 | 0.46 | +0.033 |
| (CH3)3CCN | 0.48 | 0.428 | 0.41 | +0.018 | 0.659 | 0.65 | +0.009 |
| CH3NH2 | 0.57 | 0.659 | 0.66 | -0.001 | 0.728 | 0.73 | -0.002 |
| H2N(CH2)3OH | 0.70 | 0.522 | 0.52 | +0.002 | 0.611 | 0.61 | +0.001 |
| NH3 | 1.02 | 1.087 | 1.08 | +0.007 | 1.236 | 1.22 | +0.016 |
| CH3CN | 1.02 | 0.882 | 0.89 | -0.008 | 1.081 | 1.09 | -0.009 |
| CH3SCN | 1.42 | 1.122 | 1.14 | -0.018 | 1.198 | 1.22 | -0.022 |
| ClCH2CN | 1.49 | 1.299 | 1.31 | -0.011 | 1.523 | 1.53 | -0.007 |
| NH2CHO | 1.72 | 1.998 | 2.00 | -0.002 | 2.110 | 2.11 | -0.000 |
| Cl3CCN | 1.86 | 1.499 | 1.49 | +0.009 | 1.818 | 1.82 | -0.002 |
| HNCO | 1.86 | 1.942 | 1.95 | -0.008 | 2.098 | 2.10 | -0.002 |
| ClCN | 1.87 | 1.512 | 1.52 | -0.008 | 1.744 | 1.76 | -0.016 |
| HCN | 2.22 | 2.277 | 2.28 | -0.003 | 2.427 | 2.44 | -0.013 |

Table S1 prints the values of Cl3CCN and HNCO in each other's rows; the "paper" columns above have them exchanged.

## O 1s: shifts relative to S(CH3)2O, eV

| molecule | experiment | PP-PBE here | paper | difference | PP-PBE(B3LYP) here | paper | difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| S(CH3)2O | 0.00 | 0.000 | 0.00 | +0.000 | 0.000 | 0.00 | +0.000 |
| (H2N)2CO | 0.52 | 0.495 | 0.51 | -0.015 | 0.652 | 0.65 | +0.002 |
| NH2CHO | 1.07 | 0.990 | 0.99 | +0.000 | 1.089 | 1.08 | +0.009 |
| CH3CHO | 1.95 | 1.860 | 1.85 | +0.010 | 1.957 | 1.95 | +0.007 |
| (CH3)2O | 2.36 | 2.146 | 2.15 | -0.004 | 2.246 | 2.25 | -0.004 |
| CH3OH | 2.53 | 2.485 | 2.49 | -0.005 | 2.573 | 2.58 | -0.007 |
| CH2O | 2.77 | 2.870 | 2.87 | -0.000 | 2.895 | 2.90 | -0.005 |
| COCl2 | 3.05 | 2.952 | 2.95 | +0.002 | 3.143 | 3.14 | +0.003 |
| H2O | 3.25 | 3.273 | 3.26 | +0.013 | 3.473 | 3.46 | +0.013 |
| HNCO | 3.49 | 3.626 | 3.60 | +0.026 | 3.798 | 3.77 | +0.028 |
| H2CCO | 3.58 | 3.549 | 3.56 | -0.011 | 3.657 | 3.68 | -0.023 |
| OCS | 3.63 | 3.729 | 3.73 | -0.001 | 3.904 | 3.91 | -0.006 |
| COF2 | 4.10 | 3.909 | 3.90 | +0.009 | 4.182 | 4.17 | +0.012 |
| BH3CO | 5.38 | 5.009 | 5.00 | +0.009 | 5.366 | 5.35 | +0.016 |
| CO | 5.90 | 6.019 | 6.04 | -0.021 | 6.111 | 6.14 | -0.029 |

## Summary, eV

Mean absolute error (MAE) of the shifts against experiment over the molecules of a set, the reference molecule counted as in Table S1; differences from the paper over the other molecules.

| set | molecules | method | MAE here | MAE in the paper | mean abs. difference from the paper | largest |
|---|---:|---|---:|---:|---:|---:|
| B | 9 | PP-PBE | 0.228 | 0.23 | 0.003 | 0.007 |
| B | 9 | PP-PBE(B3LYP) | 0.109 | 0.11 | 0.010 | 0.023 |
| C | 15 | PP-PBE | 0.205 | 0.20 | 0.005 | 0.016 |
| C | 15 | PP-PBE(B3LYP) | 0.175 | 0.18 | 0.008 | 0.037 |
| N | 15 | PP-PBE | 0.151 | 0.15 | 0.009 | 0.029 |
| N | 15 | PP-PBE(B3LYP) | 0.142 | 0.14 | 0.010 | 0.033 |
| O | 15 | PP-PBE | 0.108 | 0.11 | 0.009 | 0.026 |
| O | 15 | PP-PBE(B3LYP) | 0.115 | 0.12 | 0.012 | 0.029 |
| all | 54 | PP-PBE | 0.167 | 0.16 | 0.007 | 0.029 |
| all | 54 | PP-PBE(B3LYP) | 0.138 | 0.14 | 0.010 | 0.037 |

## N 1s shifts refined with alpha times the exact exchange of B3LYP (Table S2), eV

From the parts of the B3LYP energy of each case. Errors against experiment, the reference molecule counted.

| alpha | MAE here | paper | mean error here | paper |
|---:|---:|---:|---:|---:|
| 0 | 0.155 | 0.15 | +0.020 | +0.02 |
| 1 | 0.142 | 0.14 | +0.075 | +0.08 |
| 2 | 0.164 | 0.16 | +0.130 | +0.13 |
| 3 | 0.208 | 0.21 | +0.185 | +0.19 |
| 4 | 0.252 | 0.26 | +0.240 | +0.25 |
| 5 | 0.302 | 0.31 | +0.295 | +0.31 |
