# Self-consistent hybrid functionals: seven molecules against Quantum ESPRESSO

PBE0, HSE06, HSE03, B3LYP, B3LYP5, B3PW91 and Hartree-Fock, self-consistent (generalized Kohn-Sham with adaptively compressed exchange, ACE), for H2, H2O, CH4, CO, N2, benzene and triplet O2 (spin-polarized), compared with Quantum ESPRESSO 7.5 (+ libxc 7.0.0) using the same FHI98PP pseudopotentials and geometries. The quantity compared is E_hyb - E_PBE of each code at its own discretization, where both codes' PBE-level conventions cancel to first order; Fock energies and HOMOs are compared directly.

## Contents

- `<molecule>/<functional>/parsec.in` for `pbe`, `pbe0`, `hse06`, `hse03`, `b3lyp`, `b3lyp5`, `b3pw91`, `hf` (O2: with `occup.in`, 7 up and 5 down electrons), and `parsec_python.out`, the reference log of that input (one NVIDIA A100; paths shortened to repository paths).
- `H_FHIPP.DAT`, `C_FHIPP.DAT`, `N_FHIPP.DAT`, `O_FHIPP.DAT`: copies of `src/pp_generation/NCPPs/PBE/<E>/FHI/` (Troullier-Martins, FHI98PP, PBE, no core correction). The inputs find them with `--pp-dir` (one folder for all elements; the library keeps one folder per element).
- `qe/`: the Quantum ESPRESSO inputs, their settings, the UPF note and the QE numbers (`qe/README.md`).
- `libxc/run_libxc.py`: the same molecules with libxc's functionals (section "libxc's functionals").

## Run one case

From the repository root:

```powershell
python src\parsec_python\main.py examples\0d_hybrid_functionals\H2O\pbe0\parsec.in `
  --pp-dir examples\0d_hybrid_functionals --no-archive
```

The hybrid runs on the GPU with CuPy (`--device gpu`, the default when one is found); without CuPy the accelerated CLI routes the input to the reference solver, which is much slower at this grid. `parsec.out` is written beside the input; compare it with `parsec_python.out`.

Settings of every input: grid spacing 0.16 bohr, sphere = largest atom distance from the centre + 10 bohr, finite-difference order 12, `Convergence_Criterion 1e-6 Ry`, `Cubic_Spline: .true.` for every species, `Exx_Tolerance 1e-8 Ry` (default 1e-6 Ry; tight for the comparison: the default changes the energies by < 1e-6 Ry). Everything else is the default: PBE start of the Fock loop, ACE of the `States_Num` orbitals, Kronecker-product pair solver (the pair densities of the symmetry-sector orbitals in reflection-parity classes), `Exx_Inner_Criterion adaptive`, `Exx_Outer_Mixing diis`, HSE06 with w = 0.11 1/bohr; O2 keeps both spin channels in the symmetry sectors (`Spin_Symmetry_Sectors auto`).

With the adaptive inner criterion the PBE start of a spin-unpolarized hybrid stops at a residual of 1e-4 Ry. The energy its log prints for the start (the last `Total Energy` before `Fock loop`, with the note that it is the energy of a start) is therefore not a converged PBE energy: here it lies 4.0e-08 to 1.4e-05 Ry above the energy of the `pbe` input (benzene the most). E_PBE in the tables below is always the energy of the `pbe` input. The O2 start (spin-polarized) runs to `Convergence_Criterion`: its printed energy equals E_PBE to the printed digits.

## PBE level

| molecule | E_PBE this code (Ry) | E_PBE QE (Ry) | difference (Ry) | KB offset (Ry) | HOMO this code (eV) | HOMO QE (eV) |
|---|---|---|---|---|---|---|
| H2 | -2.32971479 | -2.32974716 | +3.24e-05 | 0 | -10.3764 | -10.3762 |
| H2O | -34.34610983 | -34.34601311 | -9.67e-05 | +1.75e-4 | -7.2386 | -7.2384 |
| CH4 | -16.15718582 | -16.15717314 | -1.27e-05 | -1.8e-5 | -9.4647 | -9.4647 |
| CO | -43.31671520 | -43.31651838 | -1.97e-04 | +8.7e-5 | -9.0614 | -9.0617 |
| N2 | -39.83042717 | -39.83021536 | -2.12e-04 | -4.9e-5 | -10.2959 | -10.2961 |
| C6H6 | -75.47051533 | -75.47016414 | -3.51e-04 |  | -6.3305 | -6.3306 |
| O2 (triplet) | -63.71093108 | -63.71080847 | -1.23e-04 | +2.73e-4 | -6.8729 / -11.6050 | -6.8736 / -11.6064 |

QE: 150 Ry, benzene 120 Ry (QE's H2O total at 150 Ry is 6e-5 Ry above its 250 Ry value; benzene's at 120 Ry is 6e-5 Ry above its 160 Ry value). KB offset: the change of QE's PBE energy when it uses this code's Kleinman-Bylander coefficients (below). Against QE with these coefficients the H2O difference is -2.0e-4 Ry at h 0.16 and converges to about -6e-5 Ry as h -> 0 (O converges slowest; H2 and CH4 agree within 5e-5 Ry from h 0.20 to 0.14).

## Hybrids against QE

dE = E_hyb - E_PBE (Ry), each code at its own discretization (this code: the `pbe` input); ddE = this code - QE; Fock = a_x E_x^HF of the final orbitals (Ry); HOMO in eV (O2: up / down channel). QE rows: internal `pbe0`, internal `hse` with `screening_parameter = 0.11`, libxc 402 / 475 / 401 for B3LYP / B3LYP5 / B3PW91, `hf`; HSE03 see below.

| functional | molecule | dE this code | dE QE | ddE | Fock this code | Fock QE | HOMO this code | HOMO QE |
|---|---|---|---|---|---|---|---|---|
| pbe0 | H2 | -0.00613246 | -0.00613599 | +3.5e-06 | -0.32884160 | -0.32884726 | -12.0030 | -12.0031 |
| pbe0 | H2O | +0.00543159 | +0.00540135 | +3.0e-05 | -1.96907693 | -1.96908409 | -9.0470 | -9.0473 |
| pbe0 | CH4 | -0.01578129 | -0.01579183 | +1.1e-05 | -1.47996468 | -1.47996664 | -11.0057 | -11.0058 |
| pbe0 | CO | +0.02213437 | +0.02213003 | +4.3e-06 | -2.33983044 | -2.33983251 | -10.7431 | -10.7436 |
| pbe0 | N2 | +0.02828811 | +0.02828276 | +5.4e-06 | -2.24329569 | -2.24328729 | -12.1977 | -12.1982 |
| pbe0 | C6H6 | +0.00192255 | +0.00190234 | +2.0e-05 | -5.73059354 | -5.73059923 | -7.2995 | -7.2996 |
| pbe0 | O2 (triplet) | +0.04026927 | +0.04022361 | +4.6e-05 | -3.16306350 | -3.16306389 | -9.0153 / -12.9353 | -9.0165 / -12.9364 |
| hse06 | H2 | -0.00510568 | -0.00510630 | +6.2e-07 | -0.26808897 | -0.26809615 | -11.5836 | -11.5836 |
| hse06 | H2O | +0.00762621 | +0.00760713 | +1.9e-05 | -1.72403282 | -1.72404677 | -8.6321 | -8.6324 |
| hse06 | CH4 | -0.01247447 | -0.01247594 | +1.5e-06 | -1.23633052 | -1.23633660 | -10.5917 | -10.5918 |
| hse06 | CO | +0.02493897 | +0.02494017 | -1.2e-06 | -2.03395456 | -2.03395215 | -10.3272 | -10.3276 |
| hse06 | N2 | +0.03110814 | +0.03111482 | -6.7e-06 | -1.93748241 | -1.93747648 | -11.7842 | -11.7846 |
| hse06 | C6H6 | +0.01176413 | +0.01174862 | +1.6e-05 | -4.81796759 | -4.81797872 | -6.9028 | -6.9027 |
| hse06 | O2 (triplet) | +0.04313384 | +0.04310624 | +2.8e-05 | -2.79498170 | -2.79497840 | -8.6003 / -12.5192 | -8.6017 / -12.5203 |
| hse03 | H2 | -0.04563330 | -0.04563432 | +1.0e-06 | -0.27069015 | -0.27069810 | -11.8640 | -11.8640 |
| hse03 | H2O | -0.16327978 | -0.16329488 | +1.5e-05 | -1.73367950 | -1.73369653 | -8.9094 | -8.9096 |
| hse03 | CH4 | -0.17677952 | -0.17679047 | +1.1e-05 | -1.24598219 | -1.24599087 | -10.8675 | -10.8676 |
| hse03 | CO | -0.18688697 | -0.18688571 | -1.3e-06 | -2.04587370 | -2.04587594 | -10.6061 | -10.6065 |
| hse03 | N2 | -0.18042801 | -0.18042079 | -7.2e-06 | -1.94922537 | -1.94922256 | -12.0678 | -12.0681 |
| hse03 | C6H6 | -0.60499049 | -0.60503878 | +4.8e-05 | -4.85273892 | -4.85275277 | -7.1612 | -7.1611 |
| hse03 | O2 (triplet) | -0.21600442 | -0.21604556 | +4.1e-05 | -2.80905336 | -2.80905330 | -8.8913 / -12.8013 | -8.8927 / -12.8024 |
| b3lyp | H2 | -0.02680872 | -0.02681196 | +3.2e-06 | -0.26262190 | -0.26263902 | -11.8164 | -11.8156 |
| b3lyp | H2O | -0.05079245 | -0.05074475 | -4.8e-05 | -1.57305840 | -1.57308427 | -8.8198 | -8.8190 |
| b3lyp | CH4 | -0.04323489 | -0.04323551 | +6.2e-07 | -1.18350686 | -1.18352905 | -10.7773 | -10.7768 |
| b3lyp | CO | -0.03600991 | -0.03598029 | -3.0e-05 | -1.87085538 | -1.87086635 | -10.5661 | -10.5657 |
| b3lyp | N2 | -0.01997046 | -0.01992990 | -4.1e-05 | -1.79307215 | -1.79307506 | -12.0168 | -12.0165 |
| b3lyp | C6H6 | -0.00961238 | -0.00963286 | +2.0e-05 | -4.58670899 | -4.58674653 | -7.0360 | -7.0354 |
| b3lyp | O2 (triplet) | -0.04263789 | -0.04254443 | -9.3e-05 | -2.52868134 | -2.52872814 | -8.7539 / -12.7440 | -8.7531 / -12.7432 |
| b3lyp5 | H2 | -0.01329455 | -0.01329778 | +3.2e-06 | -0.26253393 | -0.26255086 | -11.7248 | -11.7240 |
| b3lyp5 | H2O | +0.00680291 | +0.00685061 | -4.8e-05 | -1.57282084 | -1.57284672 | -8.7279 | -8.7270 |
| b3lyp5 | CH4 | +0.01252589 | +0.01252528 | +6.1e-07 | -1.18329651 | -1.18331869 | -10.6854 | -10.6848 |
| b3lyp5 | CO | +0.03569400 | +0.03572351 | -3.0e-05 | -1.87060470 | -1.87061573 | -10.4739 | -10.4735 |
| b3lyp5 | N2 | +0.05162333 | +0.05166384 | -4.1e-05 | -1.79284423 | -1.79284712 | -11.9236 | -11.9233 |
| b3lyp5 | C6H6 | +0.20188216 | +0.20186172 | +2.0e-05 | -4.58619463 | -4.58623198 | -6.9456 | -6.9449 |
| b3lyp5 | O2 (triplet) | +0.04403612 | +0.04412960 | -9.3e-05 | -2.52847706 | -2.52852392 | -8.6678 / -12.6415 | -8.6670 / -12.6407 |
| b3pw91 | H2 | -0.02627809 | -0.02628176 | +3.7e-06 | -0.26358859 | -0.26360699 | -11.7624 | -11.7617 |
| b3pw91 | H2O | -0.06939934 | -0.06941150 | +1.2e-05 | -1.57582525 | -1.57585112 | -8.7469 | -8.7463 |
| b3pw91 | CH4 | -0.07562034 | -0.07563168 | +1.1e-05 | -1.18510379 | -1.18512474 | -10.7570 | -10.7565 |
| b3pw91 | CO | -0.06880684 | -0.06879803 | -8.8e-06 | -1.87284668 | -1.87286506 | -10.4831 | -10.4829 |
| b3pw91 | N2 | -0.06052324 | -0.06051616 | -7.1e-06 | -1.79576741 | -1.79577674 | -11.9068 | -11.9065 |
| b3pw91 | C6H6 | -0.19687547 | -0.19690742 | +3.2e-05 | -4.58794965 | -4.58798564 | -7.1158 | -7.1152 |
| b3pw91 | O2 (triplet) | -0.07640795 | -0.07640513 | -2.8e-06 | -2.53131037 | -2.53135931 | -8.6537 / -12.7328 | -8.6531 / -12.7323 |
| hf | H2 | +0.06327050 | +0.06326826 | +2.2e-06 | -1.31608532 | -1.31610309 | -16.1763 | -16.1764 |
| hf | H2O | +0.58000823 | +0.58000211 | +6.1e-06 | -7.90790413 | -7.90796261 | -13.7680 | -13.7692 |
| hf | CH4 | +0.44473686 | +0.44472939 | +7.5e-06 | -5.93861804 | -5.93865321 | -14.8430 | -14.8434 |
| hf | CO | +0.78937998 | +0.78940883 | -2.9e-05 | -9.39272897 | -9.39273655 | -15.0712 | -15.0715 |
| hf | N2 | +0.82309906 | +0.82314955 | -5.0e-05 | -8.96341022 | -8.96339912 | -16.6711 | -16.6722 |
| hf | C6H6 | +2.17024244 | +2.17025095 | -8.5e-06 | -22.95270204 | -22.95275275 | -9.1338 | -9.1340 |
| hf | O2 (triplet) | +1.02027913 | +1.02026324 | +1.6e-05 | -12.70092375 | -12.70095410 | -15.0114 / -15.6670 | -15.0133 / -15.6676 |

Over the table: |ddE| <= 9.4e-05 Ry, |Fock this code - Fock QE| <= 5.9e-05 Ry, |HOMO this code - HOMO QE| <= 1.94 meV. The totals are variational and reproduce to 1e-8 Ry between runs converged to `Exx_Tolerance 1e-8 Ry`; the Fock energies and the levels are first order in the orbitals and reproduce to about 3e-5 Ry and 3e-4 eV (the last digit of a HOMO can change from one such run to the next).

Bound LUMOs (CO, N2, benzene, O2 down channel) agree within 2 meV for benzene and O2. Those of CO and N2 in the hybrids (-0.6 to -1.4 eV) are bound weakly enough to feel the sphere: +1 to +9 meV at 10 bohr of vacuum, +0.2 / +0.8 meV (PBE0) with 4 bohr more. The LUMOs of H2, H2O, CH4 are diffuse domain states in both codes and those of HF are box states: not compared.

## Known offsets

**PARSEC's KB quadrature.** This code (as Fortran PARSEC) integrates the Kleinman-Bylander energies E_l = int u_l^2 (V_l - V_loc) dr with the trapezoid in r on the logarithmic mesh, which exceeds the converged integral by sinh(dx)/dx - 1 = 9.9e-5 relative; QE uses Simpson's rule. This is a constant that does not vanish with h. Measured in QE with this code's coefficients, it shifts E_PBE by the values in the PBE table. It mostly cancels in E_hyb - E_PBE, but not completely when the hybrid changes how the occupied orbitals see the projectors: -2.3e-5 Ry (N2 HF), -2.1e-5 (CO HF), -1.6e-5 (H2O HF), -1.1e-5 (O2 PBE0), -7e-6 (N2 PBE0), <= 3e-6 for B3LYP.

**QE's cutoff error for LYP (and for O).** QE's E_hyb - E_PBE at 150 Ry is not converged to 1e-5 Ry for the LYP functionals and for oxygen. The same QE build at higher cutoffs (L = 24 bohr, ecutfock = 2 ecutwfc):

| molecule | functional | QE 150 Ry | QE at ecut | ecut (Ry) | this code | this code - QE 150 | this code - QE at ecut |
|---|---|---|---|---|---|---|---|
| H2O | pbe0 | +0.00540135 | +0.00542384 | 300 | +0.00543159 | +3.0e-05 | +7.8e-06 |
| H2O | b3lyp | -0.05074475 | -0.05078664 | 300 | -0.05079245 | -4.8e-05 | -5.8e-06 |
| N2 | pbe0 | +0.02828276 | +0.02829726 | 250 | +0.02828811 | +5.4e-06 | -9.2e-06 |
| N2 | b3lyp | -0.01992990 | -0.01995836 | 250 | -0.01997046 | -4.1e-05 | -1.2e-05 |
| N2 | hf | +0.82314955 | +0.82314394 | 250 | +0.82309906 | -5.0e-05 | -4.5e-05 |
| O2 (triplet) | pbe0 | +0.04022361 | +0.04026462 | 250 | +0.04026927 | +4.6e-05 | +4.7e-06 |
| O2 (triplet) | b3lyp | -0.04254443 | -0.04262701 | 250 | -0.04263789 | -9.3e-05 | -1.1e-05 |
| C6H6 | pbe0 | +0.00190234 | +0.00190887 | 160 (QE 120 above) | +0.00192255 | +2.0e-05 | +1.4e-05 |
| C6H6 | hse06 | +0.01174862 | +0.01175731 | 160 (QE 120 above) | +0.01176413 | +1.6e-05 | +6.8e-06 |
| C6H6 | hse03 | -0.60503878 | -0.60502880 | 160 (QE 120 above) | -0.60499049 | +4.8e-05 | +3.8e-05 |

With both corrections (higher cutoff and this code's KB coefficients in QE) the remaining differences of the cases above are within 2.2e-5 Ry (N2 HF) and mostly below 1e-5. B3LYP5 has the same differences as B3LYP to 1e-7 Ry; B3PW91, with the same exchange and PW91 correlation, does not show the LYP effect. On our side, E_hyb - E_PBE changes by at most 1.3e-5 Ry between h 0.16, 0.14 and 0.12 bohr (H2O, N2).

**B3PW91 constants.** The PW91 and PW92 pieces of this code keep Fortran PARSEC's constants, which differ from libxc's in the last digits (PW92 A, PW91 nu and C_x, the Rasolt-Geldart rs^3 term). On the converged densities libxc's constants change the B3PW91 energy by -0.3e-6 (H2) to -2.6e-6 Ry (O2).

**B3LYP definitions.** `b3lyp` is B3LYP with VWN-RPA (libxc HYB_GGA_XC_B3LYP, Gaussian's), `b3lyp5` with VWN5. QE's internal `B3LYP` uses VWN5 and its own B88/LYP, which differ from libxc's by about 1e-4 Ry for H2O; QE's `B3LYP-V1R` interpolates VWN-RPA in spin differently (8e-4 Ry for O2). The references are therefore QE with libxc 402 and 475.

## libxc's functionals

The same inputs with the functionals of libxc (the optional library, `LibXC_Functional`; see "Functionals of
libxc" in `src/parsec_python/README.md`): `libxc/run_libxc.py` copies a case, replaces its `Correlation_Type` by
the libxc keyword and runs `main.py` (one A100, libxc 7.1.2 CUDA build, the libxc functional evaluated on the
GPU, 2026-11-07):

```bash
PARSEC_LIBXC_LIBRARY=.../libxc.so python examples/0d_hybrid_functionals/libxc/run_libxc.py --out OUT
```

**The code's functionals and libxc's** (total energies, Ry; libxc - this code):

| molecule | `pbe` | `gga_x_pbe gga_c_pbe` | `b3lyp` | `hyb_gga_xc_b3lyp` | `pbe0` | `hyb_gga_xc_pbeh` |
|---|---|---|---|---|---|---|
| H2 | -2.32971479 | -1e-8 | -2.35652351 | 0 | -2.33584725 | -1e-8 |
| H2O | -34.34610983 | -8e-8 | -34.39690228 | 0 | -34.34067824 | -8e-8 |
| CH4 | -16.15718582 | -6e-8 | -16.20042071 | 0 | -16.17296711 | -6e-8 |
| CO | -43.31671520 | -9e-8 | -43.35272511 | 0 | -43.29458083 | -9e-8 |
| N2 | -39.83042717 | -9e-8 | -39.85039763 | 0 | -39.80213906 | -9e-8 |
| O2 (triplet) | -63.71093108 | -1.3e-7 | -63.75356897 | 0 | | |

B3LYP is the same functional in both (also with spin: O2), to the printed digits, and so are the exchange
energies `a_x E_x`. The PBE of this code differs from libxc's by the PW92 `A` of its correlation
(`(1 - ln 2)/pi^2` against libxc's 0.0310907): 1e-8 to 1.3e-7 Ry here, and by the same in PBE0.

**libxc's HSE06 (428) and HSE03 (427) against Quantum ESPRESSO with libxc** (QE's `hse06_libxc` and
`hse03_libxc` rows, `qe/README.md`): dE = E_hyb - E_PBE (this code: the `pbe` input), Fock = `a_x E_x` (Ry).

| functional | molecule | dE this code | dE QE | ddE | Fock this code | Fock QE | dFock |
|---|---|---|---|---|---|---|---|
| hse06 (libxc 428) | H2 | -0.00631429 | -0.00631791 | +3.6e-06 | -0.26801282 | -0.26801734 | +4.5e-06 |
| hse06 (libxc 428) | H2O | +0.00751581 | +0.00749483 | +2.1e-05 | -1.72360290 | -1.72361325 | +1.0e-05 |
| hse06 (libxc 428) | CH4 | -0.01277033 | -0.01278614 | +1.6e-05 | -1.23611075 | -1.23611379 | +3.0e-06 |
| hse06 (libxc 428) | CO | +0.02581786 | +0.02581579 | +2.1e-06 | -2.03362500 | -2.03362469 | -3.1e-07 |
| hse06 (libxc 428) | N2 | +0.03241038 | +0.03241806 | -7.7e-06 | -1.93709788 | -1.93708486 | -1.3e-05 |
| hse03 (libxc 427) | H2 | -0.04684184 | -0.04684600 | +4.2e-06 | -0.27061363 | -0.27061929 | +5.7e-06 |
| hse03 (libxc 427) | H2O | -0.16338835 | -0.16340967 | +2.1e-05 | -1.73325097 | -1.73326301 | +1.2e-05 |
| hse03 (libxc 427) | CH4 | -0.17707994 | -0.17709741 | +1.7e-05 | -1.24576303 | -1.24576806 | +5.0e-06 |
| hse03 (libxc 427) | CO | -0.18600764 | -0.18601185 | +4.2e-06 | -2.04554638 | -2.04554848 | +2.1e-06 |
| hse03 (libxc 427) | N2 | -0.17912561 | -0.17911944 | -6.2e-06 | -1.94884254 | -1.94883094 | -1.2e-05 |

The semilocal part from libxc and the screened exact exchange of this code reproduce QE's libxc hybrids within
2.1e-5 Ry, the level of the code's own HSE06 against QE's (table above). libxc's HSE06 differs from the code's
(QE's form, `hse06` above) by -1.1e-4 (H2O) to +1.3e-3 Ry (N2): libxc's full-range wPBEh(w = 0) exchange and
its rescaling of s.

## Why `Cubic_Spline` is set

Without it, this code (as PARSEC) interpolates r V(r) linearly between the samples of the FHI logarithmic mesh (dr/r = 2.4 %). Against the smooth potential that QE Fourier-transforms this shifts the energy by about +2.2e-3 Ry per O, +4e-4 per C and +1.5e-5 per H, makes the dependence on h erratic, and lowers the H2O HOMO by 3 to 4.5 meV. `Cubic_Spline: .true.` (PARSEC's spline of the radial table) removes it (1e-7 Ry).

## HSE03

`hse03` is HSE in QE's (= VASP's) form with w_HF = 0.15/sqrt(2) = 0.106066 and w_PBE = 0.15*2^(1/3) = 0.188988 1/bohr. QE's internal HSE has one w for both parts, and libxc's HSE03 (427) writes the full-range exchange as wPBEh(w = 0) instead of PBE and rescales s differently. The reference is therefore derived: hse03_libxc + (hse06 - hse06_libxc), which is HSE03 in QE's form if the libxc-minus-QE form difference does not depend on w. The s-rescaling part of that difference does depend on w: on our densities it makes the derived value 2e-6 (H2) to 9e-6 Ry (benzene) too high. The derived Fock energies and HOMOs are estimates made the same way. Benzene (+3.8e-5 Ry against QE at 160 Ry, HSE06 +6.8e-6) is the largest HSE03 difference; the two functionals share all code except the two w values, so the excess is attributed to the derived reference.

## Timings (indicative)

Wall time of `main.py` (program wall, PBE start included) on one A100 of a shared node, against QE on 4 MPI ranks of a laptop (different hardware):

| molecule | grid points | PBE (s) | hybrids (s) | HF (s) | QE, hybrids and HF (s) |
|---|---|---|---|---|---|
| H2 | 1,252,368 | 6 | 9-17 | 11 | 53-158 |
| H2O | 1,568,208 | 12 | 14-23 | 22 | 86-294 |
| CH4 | 1,788,656 | 11 | 15-27 | 24 | 95-283 |
| CO | 1,387,496 | 10 | 15-22 | 18 | 114-299 |
| N2 | 1,376,312 | 8 | 12-21 | 14 | 99-258 |
| C6H6 | 3,241,288 | 13 | 33-52 | 38 | 404-744 |
| O2 (triplet) | 1,413,968 | 14 | 25-31 | 29 | 257-654 |

Outer Fock iterations to an estimated exchange error below 1e-8 Ry: 3-4 for PBE0, HSE and B3*, 6-7 for HF (QE, whose loop runs to 1e-10 Ry and starts the libxc hybrids from the functional without its exact exchange and HF from Slater exchange: 4-6 and 10-16). The table gives outer iterations / inner SCF iterations of the Fock loop (summed over its outer iterations); the PBE start takes 13-29 SCF iterations before them, the `pbe` input 20-44. The exact exchange itself (pair solves and ACE builds) is 0.3-21.8 s of a run.

| molecule | pbe0 | hse06 | hse03 | b3lyp | b3lyp5 | b3pw91 | hf |
|---|---|---|---|---|---|---|---|
| H2 | 3 / 39 | 3 / 40 | 3 / 27 | 3 / 26 | 3 / 21 | 3 / 27 | 6 / 38 |
| H2O | 4 / 35 | 3 / 25 | 3 / 29 | 3 / 23 | 3 / 37 | 4 / 30 | 7 / 62 |
| CH4 | 4 / 29 | 4 / 47 | 4 / 42 | 4 / 36 | 4 / 37 | 4 / 32 | 7 / 75 |
| CO | 4 / 39 | 4 / 44 | 4 / 44 | 4 / 32 | 4 / 31 | 4 / 31 | 7 / 44 |
| N2 | 4 / 28 | 4 / 44 | 4 / 33 | 4 / 38 | 4 / 35 | 4 / 30 | 7 / 45 |
| C6H6 | 4 / 41 | 4 / 51 | 4 / 41 | 4 / 48 | 4 / 46 | 4 / 44 | 7 / 57 |
| O2 (triplet) | 4 / 43 | 4 / 52 | 4 / 44 | 4 / 43 | 4 / 47 | 4 / 39 | 7 / 52 |
