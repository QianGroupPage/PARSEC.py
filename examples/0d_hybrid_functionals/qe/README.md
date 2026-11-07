# Quantum ESPRESSO side

Inputs `<molecule>_<row>.in` of the reference values in `../README.md` (QE 7.5 with libxc 7.0.0, pw.x, 4 MPI ranks). Run with the UPF files in `./pseudo` (below), e.g. `mpirun -np 4 pw.x -in H2O_pbe0.in`.

## Pseudopotentials

The UPF files are the FHI98PP tables of `src/pp_generation/NCPPs/PBE/<E>/FHI/` in Kleinman-Bylander form with the same local channel (d for C, N, O; s for H) and one projector per l: beta_l = u_l (V_l - V_loc), D_l = 1/E_l. QE 7.5's `upfconv.x` lists `.fhi` but no longer reads it, so the references used QE's legacy FHI98PP-to-UPF conversion (QE's `read_fhi` and `write_upf` routines). The repository's `src/pp_generation/NCPPs/PBE/<E>/UPF/<e>_pbe_tm.UPF` files carry the same tables (they differ in a header comment and the label of the unused d wavefunction); copy them into `qe/pseudo/`. The UPF tables equal the FHI tables as this code reads them to 1e-15.

## Settings

- Same geometries (bohr) as the parsec inputs, centred in a cubic box; `assume_isolated = 'mt'`, Gamma only, `occupations = 'fixed'`; O2 `nspin = 2`, `tot_magnetization = 2`; `conv_thr = 1e-10` Ry, which also stops the exact-exchange loop.
- `ecutwfc = 150` Ry and `ecutfock = 300` Ry; benzene 120 / 240 Ry (`ecutfock = 2 ecutwfc` equals the default 4 ecutwfc to 1e-8 Ry for H2O).
- Box: L = 24 bohr for PBE and the Coulomb-kernel hybrids, 28 bohr for the erfc kernel (HSE), 32 bohr for benzene.
- Exact exchange: `nqx1 = nqx2 = nqx3 = 1`, `x_gamma_extrapolation = .false.`. Coulomb kernel (PBE0, B3*, HF): `exxdiv_treatment = 'vcut_spherical'` (kernel truncated at 0.49 L; H2O PBE0 within 6e-6 Ry from L = 20 to 32). erfc kernel (HSE): `exxdiv_treatment = 'gygi-baldereschi'`, because `vcut_spherical` returns the bare truncated Coulomb kernel before the erfc branch and would turn HSE into a PBE0-like functional; the images then decay like erfc(w L)/L (H2O within 5e-6 Ry at L = 28).
- Functionals: `pbe0`; `hse` with `screening_parameter = 0.11` (QE's default is 0.106; one w for the exact and the PBE part); libxc through QE for B3LYP (`XC-000I-000I-402L-000I-000I-000I`), B3LYP5 (475), B3PW91 (401), HSE06 (428, `_hse06_libxc`) and HSE03 (427, `_hse03_libxc`); `hf`. `_internal` inputs: QE's own `b3lyp-v1r` (H2O, O2) and `b3lyp` (VWN5, H2O), listed for comparison only.
- ACE is QE's default (off for H2, one occupied orbital). QE starts the hybrids from a semilocal SCF without exact exchange (PBE; for the libxc hybrids the libxc functional without its exact part; HF from Slater exchange).

## QE numbers

E and E - E_PBE in Ry (E_PBE of the same box and cutoff), Fock = `+ Fock energy (ACE)` = a_x E_x^HF in Ry, HOMO and LUMO in eV (O2: up / down channel; LUMO of O2 = down channel), outer = exact-exchange iterations, dexx < 1e-10 Ry at the end. hse03_derived = hse03_libxc + (hse06 - hse06_libxc) (not a calculation).

| molecule | row | E (Ry) | E - E_PBE (Ry) | Fock (Ry) | HOMO (eV) | LUMO (eV) | outer |
|---|---|---|---|---|---|---|---|
| H2 | pbe | -2.32974716 | +0.00000000 |  | -10.3762 | -0.0904 | 0 |
| H2 | pbe0 | -2.33588315 | -0.00613599 | -0.32884726 | -12.0031 | -0.0365 | 4 |
| H2 | hse06 | -2.33485339 | -0.00510630 | -0.26809615 | -11.5836 | -0.0467 | 4 |
| H2 | hse06_libxc | -2.33606500 | -0.00631791 | -0.26801734 | -11.5971 | -0.0494 | 5 |
| H2 | hse03_libxc | -2.37659309 | -0.04684600 | -0.27061929 | -11.8775 | -0.0526 | 5 |
| H2 | hse03_derived | -2.37538148 | -0.04563432 |  | -11.8640 | -0.0499 |  |
| H2 | b3lyp_libxc | -2.35655912 | -0.02681196 | -0.26263902 | -11.8156 | -0.0841 | 5 |
| H2 | b3lyp5_libxc | -2.34304494 | -0.01329778 | -0.26255086 | -11.7240 | -0.0732 | 5 |
| H2 | b3pw91 | -2.35602892 | -0.02628176 | -0.26360699 | -11.7617 | -0.0507 | 5 |
| H2 | hf | -2.26647890 | +0.06326826 | -1.31610309 | -16.1764 | 0.0533 | 10 |
| H2O | pbe | -34.34601311 | +0.00000000 |  | -7.2384 | -0.9491 | 0 |
| H2O | pbe0 | -34.34061176 | +0.00540135 | -1.96908409 | -9.0473 | -0.4573 | 5 |
| H2O | hse06 | -34.33840349 | +0.00760713 | -1.72404677 | -8.6324 | -0.6489 | 4 |
| H2O | hse06_libxc | -34.33851579 | +0.00749483 | -1.72361325 | -8.6308 | -0.6588 | 6 |
| H2O | hse03_libxc | -34.50942029 | -0.16340967 | -1.73326301 | -8.9080 | -0.7461 | 6 |
| H2O | hse03_derived | -34.50930799 | -0.16329488 |  | -8.9096 | -0.7362 |  |
| H2O | b3lyp_libxc | -34.39675786 | -0.05074475 | -1.57308427 | -8.8190 | -0.6857 | 6 |
| H2O | b3lyp (internal) | -34.39665237 | -0.05063926 | -1.57315565 | -8.8160 | -0.6110 | 4 |
| H2O | b3lyp5_libxc | -34.33916250 | +0.00685061 | -1.57284672 | -8.7270 | -0.6347 | 6 |
| H2O | b3lyp5 (internal) | -34.33905653 | +0.00695658 | -1.57291859 | -8.7241 | -0.5596 | 4 |
| H2O | b3pw91 | -34.41542461 | -0.06941150 | -1.57585112 | -8.7463 | -0.4549 | 6 |
| H2O | hf | -33.76601100 | +0.58000211 | -7.90796261 | -13.7692 | 0.0438 | 14 |
| CH4 | pbe | -16.15717314 | +0.00000000 |  | -9.4647 | -0.4580 | 0 |
| CH4 | pbe0 | -16.17296497 | -0.01579183 | -1.47996664 | -11.0058 | -0.1944 | 5 |
| CH4 | hse06 | -16.16965234 | -0.01247594 | -1.23633660 | -10.5918 | -0.2900 | 5 |
| CH4 | hse06_libxc | -16.16996254 | -0.01278614 | -1.23611379 | -10.5991 | -0.2986 | 6 |
| CH4 | hse03_libxc | -16.33427381 | -0.17709741 | -1.24576806 | -10.8749 | -0.3443 | 6 |
| CH4 | hse03_derived | -16.33396361 | -0.17679047 |  | -10.8676 | -0.3357 |  |
| CH4 | b3lyp_libxc | -16.20040865 | -0.04323551 | -1.18352905 | -10.7768 | -0.3228 | 6 |
| CH4 | b3lyp5_libxc | -16.14464786 | +0.01252528 | -1.18331869 | -10.6848 | -0.2887 | 6 |
| CH4 | b3pw91 | -16.23280482 | -0.07563168 | -1.18512474 | -10.7565 | -0.1883 | 6 |
| CH4 | hf | -15.71244375 | +0.44472939 | -5.93865321 | -14.8434 | 0.1081 | 15 |
| CO | pbe | -43.31651838 | +0.00000000 |  | -9.0617 | -2.0048 | 0 |
| CO | pbe0 | -43.29438835 | +0.02213003 | -2.33983251 | -10.7436 | -0.7814 | 6 |
| CO | hse06 | -43.29157645 | +0.02494017 | -2.03395215 | -10.3276 | -1.1643 | 5 |
| CO | hse06_libxc | -43.29070083 | +0.02581579 | -2.03362469 | -10.3323 | -1.1718 | 6 |
| CO | hse03_libxc | -43.50252847 | -0.18601185 | -2.04554848 | -10.6112 | -1.3973 | 6 |
| CO | hse03_derived | -43.50340409 | -0.18688571 |  | -10.6065 | -1.3898 |  |
| CO | b3lyp_libxc | -43.35249867 | -0.03598029 | -1.87086635 | -10.5657 | -1.0756 | 6 |
| CO | b3lyp5_libxc | -43.28079487 | +0.03572351 | -1.87061573 | -10.4735 | -0.9883 | 6 |
| CO | b3pw91 | -43.38531641 | -0.06879803 | -1.87286506 | -10.4829 | -1.0321 | 6 |
| CO | hf | -42.52710955 | +0.78940883 | -9.39273655 | -15.0715 | 0.1464 | 16 |
| N2 | pbe | -39.83021536 | +0.00000000 |  | -10.2961 | -1.9659 | 0 |
| N2 | pbe0 | -39.80193260 | +0.02828276 | -2.24328729 | -12.1982 | -0.6336 | 5 |
| N2 | hse06 | -39.79909865 | +0.03111482 | -1.93747648 | -11.7846 | -1.0235 | 5 |
| N2 | hse06_libxc | -39.79779541 | +0.03241806 | -1.93708486 | -11.7758 | -1.0338 | 6 |
| N2 | hse03_libxc | -40.00933291 | -0.17911944 | -1.94883094 | -12.0593 | -1.2727 | 6 |
| N2 | hse03_derived | -40.01063615 | -0.18042079 |  | -12.0681 | -1.2624 |  |
| N2 | b3lyp_libxc | -39.85014526 | -0.01992990 | -1.79307506 | -12.0165 | -0.9671 | 5 |
| N2 | b3lyp5_libxc | -39.77855152 | +0.05166384 | -1.79284712 | -11.9233 | -0.8773 | 5 |
| N2 | b3pw91 | -39.89073152 | -0.06051616 | -1.79577674 | -11.9065 | -0.9208 | 5 |
| N2 | hf | -39.00706581 | +0.82314955 | -8.96339912 | -16.6722 | 0.1420 | 12 |
| C6H6 | pbe | -75.47016414 | +0.00000000 |  | -6.3306 | -1.2382 | 0 |
| C6H6 | pbe0 | -75.46826180 | +0.00190234 | -5.73059923 | -7.2996 | -0.3279 | 6 |
| C6H6 | hse06 | -75.45841552 | +0.01174862 | -4.81797872 | -6.9027 | -0.6993 | 5 |
| C6H6 | hse06_libxc | -75.45048723 | +0.01967691 | -4.81781483 | -6.9085 | -0.7088 | 6 |
| C6H6 | hse03_libxc | -76.06727463 | -0.59711049 | -4.85258888 | -7.1669 | -0.9354 | 6 |
| C6H6 | hse03_derived | -76.07520292 | -0.60503878 |  | -7.1611 | -0.9259 |  |
| C6H6 | b3lyp_libxc | -75.47979700 | -0.00963286 | -4.58674653 | -7.0354 | -0.4755 | 6 |
| C6H6 | b3lyp5_libxc | -75.26830242 | +0.20186172 | -4.58623198 | -6.9449 | -0.3873 | 6 |
| C6H6 | b3pw91 | -75.66707156 | -0.19690742 | -4.58798564 | -7.1152 | -0.5054 | 6 |
| C6H6 | hf | -73.29991319 | +2.17025095 | -22.95275275 | -9.1340 | 0.1628 | 14 |
| O2 (triplet) | pbe | -63.71080847 | +0.00000000 |  | -6.8736 / -11.6064 | -4.3938 | 0 |
| O2 (triplet) | pbe0 | -63.67058486 | +0.04022361 | -3.16306389 | -9.0165 / -12.9364 | -2.8337 | 6 |
| O2 (triplet) | hse06 | -63.66771564 | +0.04310624 | -2.79497840 | -8.6017 / -12.5203 | -3.2304 | 6 |
| O2 (triplet) | hse06_libxc | -63.66737380 | +0.04344808 | -2.79452105 | -8.5981 / -12.5277 | -3.2476 | 6 |
| O2 (triplet) | hse03_libxc | -63.92651219 | -0.21569031 | -2.80859595 | -8.8891 / -12.8098 | -3.5013 | 6 |
| O2 (triplet) | hse03_derived | -63.92685403 | -0.21604556 |  | -8.8927 | -3.4841 |  |
| O2 (triplet) | b3lyp_libxc | -63.75335290 | -0.04254443 | -2.52872814 | -8.7531 / -12.7432 | -3.3030 | 6 |
| O2 (triplet) | b3lyp (internal) | -63.75416032 | -0.04335185 | -2.52875727 | -8.7644 / -12.7351 | -3.2938 | 5 |
| O2 (triplet) | b3lyp5_libxc | -63.66667887 | +0.04412960 | -2.52852392 | -8.6670 / -12.6407 | -3.1994 | 6 |
| O2 (triplet) | b3pw91 | -63.78721360 | -0.07640513 | -2.53135931 | -8.6531 / -12.7323 | -3.2172 | 6 |
| O2 (triplet) | hf | -62.69054523 | +1.02026324 | -12.70095410 | -15.0133 / -15.6676 | 0.1271 | 16 |

E - E_PBE of the HSE rows (box 28 bohr) uses the PBE energy of the same box (`<molecule>_pbe_L28.in`); benzene uses 32 bohr for all rows. Cutoff convergence of E - E_PBE and the effect of this code's KB coefficients: see `../README.md`.
