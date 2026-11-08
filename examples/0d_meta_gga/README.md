# Meta-GGAs and meta-GGA hybrids: six molecules against Quantum ESPRESSO

SCAN (this code's native SCAN), r2SCAN and TPSS (libxc) and the meta-GGA hybrids TPSSh and r2SCAN0 (libxc's
semilocal part with this code's exact exchange), self-consistent, for H2O, CH4, NH3, N2, CO and triplet O2
(spin-polarized), compared with Quantum ESPRESSO 7.5 (+ libxc 7.0.0) using the same FHI98PP pseudopotentials and
geometries; forces of a distorted water molecule. The quantity compared is E_mGGA - E_PBE of each code at its own
discretization, where both codes' PBE-level conventions cancel to first order; HOMOs, exact-exchange energies and
forces are compared directly. Bulk Si is in [`../3d_meta_gga`](../3d_meta_gga/README.md).

## Contents

- `<molecule>/<variant>/parsec.in` (O2: with `occup.in`, 7 up and 5 down electrons) and `parsec_python.out`, the
  reference log of that input (one NVIDIA A100; paths shortened to repository paths, the libxc library as
  `$PARSEC_LIBXC_LIBRARY`). Variants: `pbe`, `scan`, `r2scan`, `tpss` for the six molecules; `tpssh`, `r2scan0` for
  the five closed shells; `pbe_h013`, `r2scan_h013` (grid spacing 0.13 bohr) for H2O and O2; `H2O_dist/pbe`,
  `H2O_dist/r2scan`: water with one O-H bond stretched to 1.10 A, with forces.
- Pseudopotentials (copies of the project's library `src/pp_generation/NCPPs/PBE`, found with `--pp-dir`):

  | file | from | used by |
  |---|---|---|
  | `H_FHIPP.DAT`, `C_FHIPP.DAT`, `N_FHIPP.DAT`, `O_FHIPP.DAT` | `NCPPs/PBE/<element>/FHI` (Troullier-Martins, PBE, no core correction) | the inputs here, `python -m parsec_python.acceleration.benchmarks.meta_gga_check`, `tests/unit/test_meta_gga_scf.py` |
  | `H_POTRE.DAT`, `Li_POTRE.DAT` | `NCPPs/PBE/<element>/PARSEC` | `tests/unit/test_meta_gga_scf.py` (LiH: Li has a nonlinear core correction) |

- `qe/`: the Quantum ESPRESSO inputs of the compared rows, their settings and QE's cutoff convergence
  (`qe/README.md`).
- `run_examples.py` (one process per input, `main.py --gpus 1`; `--install` copies the logs of a run folder into the
  example) and `collect_results.py` (the tables below from the logs).

## Run

r2SCAN, TPSS and the hybrids are libxc's: build or install libxc and point `PARSEC_LIBXC_LIBRARY` at it ("Functionals
of libxc" in `src/parsec_python/README.md`; the logs here used libxc 7.1.2, CUDA build, evaluated on the GPU). From
the repository root:

```bash
export PARSEC_LIBXC_LIBRARY=.../libxc-7.1.2-cuda-sm80/lib/libxc.so
python src/parsec_python/main.py examples/0d_meta_gga/H2O/r2scan/parsec.in --pp-dir examples/0d_meta_gga --no-archive
python examples/0d_meta_gga/run_examples.py --out RUNS          # every input, logs in RUNS/<molecule>/<variant>/
python examples/0d_meta_gga/collect_results.py [--runs RUNS]
```

`parsec.out` is written beside the input; compare it with `parsec_python.out`. SCAN needs no library.

Settings of every input (those of `../0d_hybrid_functionals`): grid spacing 0.16 bohr, sphere = largest atom distance
from the centre + 10 bohr, `Expansion_Order` 12, `Convergence_Criterion` 1e-6 Ry, `Max_Iter` 200, `Cubic_Spline` for
every species; the hybrids `Exx_Tolerance` 1e-8 Ry (default 1e-6 Ry, tight for the comparison). Everything else is
the default.

### The functional in the input

Three spellings, each used here:

```text
Correlation_Type: scan                                  # this code's SCAN (scan/)
LibXC_Functional: mgga_x_r2scan mgga_c_r2scan           # libxc by name (r2scan/, and the hybrids:
LibXC_Functional: hyb_mgga_xc_tpssh                     #   tpssh/, r2scan0/)
Use_LibXC: .true.                                       # PARSEC 1.5's keywords, libxc by number (tpss/):
Func_Id_X: 202                                          #   MGGA_X_TPSS
Func_Id_C: 231                                          #   MGGA_C_TPSS
```

`LibXC_Functional: mgga_x_tpss mgga_c_tpss` is the same functional as the 1.5 form (the run's settings are equal).
In 1.5's form an exchange-correlation functional in `Func_Id_X` (`457` for TPSSh) gets 1.5's default `Func_Id_C`
(LDA correlation) added, as 1.5 does, with a warning; `Func_Id_C: 0` leaves it out. The hybrids here are therefore
written with `LibXC_Functional`. `python -m parsec_python.V_xc.libxc --list` lists libxc's functionals.

### Defaults and how to switch them

Each log names what it did under "Correlation data".

- **Density gradient.** `auto`: the orbital form `g = sum_n f_n 2 psi_n (G psi_n)` for functionals smooth across
  `tau = tau_W` (SCAN, r2SCAN, r2SCAN0: "density gradient: orbital form") and the stencil gradient of the density for
  those with a kink there (TPSS, TPSSh: "density form"). `PARSEC_META_GGA_GRADIENT=orbital` or `density` forces one.
- **Floors of `tau`.** libxc meta-GGAs floor `tau` of each channel at `LibXC_Tau_Threshold` (1e-12 Hartree/bohr^3:
  "tau floor 1e-12"); TPSS-type functionals also at `tau_W` with the exact derivatives ("tau >= tau_W floor (Fermi-hole
  curvature; the functional has a kink at tau_W)"), so that all their components see `z = tau_W/tau <= 1`.
  `LibXC_Enforce_FHC: .false.` evaluates them at the `tau` of the orbitals, `.true.` floors any libxc meta-GGA. libxc's
  own clamp (`sigma -> 8 n tau`, not carried into the derivatives) is never used; Quantum ESPRESSO switches it off as
  well. The native SCAN has no floor (`PARSEC_SCAN_TAU_FLOOR=weizsaecker` adds the `tau_W` one).
- **First iteration and mixing.** The first iteration is PBE of the start density (`vtau = 0`), as PARSEC 1.5 starts
  its libxc meta-GGAs; the mixer mixes `(V, vtau[, W])` of all channels jointly with Anderson's least squares
  weighted by the density (`PARSEC_META_GGA_ANDERSON=weighted`; `plain` for PARSEC's metric). The filter's upper bound
  takes in Gershgorin's bound of the orbital term (`PARSEC_META_GGA_FILTER_BOUND=gershgorin`).
- **Meta-GGA hybrids** run the Fock loop of the hybrids (PBE start, ACE, `Exx_*` keywords) with meta-GGA SCFs as
  inner loops; libxc gives the exact-exchange fraction (TPSSh 0.10, r2SCAN0 0.25).

All switches: "Meta-GGA functionals" and "Functionals of libxc" in `src/parsec_python/README.md`.

## PBE level

QE at 300 Ry (N2, CO: 150 Ry); HOMO in eV (O2: up / down channel).

| molecule | E_PBE this code (Ry) | E_PBE QE (Ry) | QE cutoff (Ry) | difference (Ry) | HOMO this code (eV) | HOMO QE (eV) | dHOMO (meV) |
|---|---|---|---|---|---|---|---|
| H2O | -34.34610983 | -34.34607194 | 300 | -3.79e-05 | -7.2386 | -7.2383 | -0.3 |
| CH4 | -16.15718582 | -16.15718789 | 300 | +2.07e-06 | -9.4647 | -9.4644 | -0.3 |
| NH3 | -23.46693353 | -23.46695167 | 300 | +1.81e-05 | -6.1612 | -6.1604 | -0.8 |
| N2 | -39.83042717 | -39.83021459 | 150 | -2.13e-04 | -10.2959 | -10.2959 | +0.0 |
| CO | -43.31671520 | -43.31651727 | 150 | -1.98e-04 | -9.0614 | -9.0616 | +0.2 |
| O2 | -63.71093108 | -63.71092564 | 300 | -5.44e-06 | -6.8729 / -11.6050 | -6.8741 / -11.6063 | +1.2 / +1.3 |

The PBE inputs equal those of `../0d_hybrid_functionals` (its README explains the remaining offsets: PARSEC's
Kleinman-Bylander quadrature and QE's cutoff at 150 Ry).

## Meta-GGAs against QE

E - E_PBE in mRy, each code against its own PBE (QE: at the cutoff of the column, 300 Ry where QE's convergence rows
exist, TPSS of H2O and O2 250 Ry, else 150 Ry; `qe/README.md`); HOMO in eV (O2: up / down). "Max_Iter": the run
stopped at `Max_Iter` 200 with the energy settled (see "SCAN" below).

| molecule | functional | E - E_PBE this code (mRy) | E - E_PBE QE (mRy) | QE cutoff (Ry) | this code - QE (mRy) | HOMO this code (eV) | HOMO QE (eV) | dHOMO (meV) | SCF iterations | gradient form |
|---|---|---|---|---|---|---|---|---|---|---|
| H2O | SCAN | +1.397 | +1.762 | 300 | -0.365 | -7.5580 | -7.5973 | +39.3 | 45 | orbital |
| H2O | r2SCAN | -2.132 | -2.024 | 300 | -0.109 | -7.5678 | -7.5688 | +1.0 | 31 | orbital |
| H2O | TPSS | -11.075 | -11.083 | 250 | +0.008 | -7.3155 | -7.3158 | +0.3 | 43 | density |
| CH4 | SCAN (Max_Iter) | -4.046 | -3.924 | 300 | -0.122 | -9.8254 | -9.8252 | -0.2 | 200 | orbital |
| CH4 | r2SCAN | -1.511 | -1.443 | 300 | -0.067 | -9.8132 | -9.8132 | +0.0 | 29 | orbital |
| CH4 | TPSS | -39.429 | -39.357 | 150 | -0.072 | -9.6569 | -9.6566 | -0.3 | 28 | density |
| NH3 | SCAN (Max_Iter) | +6.216 | +6.516 | 300 | -0.300 | -6.5035 | -6.4947 | -8.8 | 200 | orbital |
| NH3 | r2SCAN | +4.850 | +4.822 | 300 | +0.028 | -6.4626 | -6.4625 | -0.1 | 62 | orbital |
| NH3 | TPSS | -17.648 | -17.640 | 150 | -0.008 | -6.2283 | -6.2282 | -0.1 | 45 | density |
| N2 | SCAN | +38.779 | +39.273 | 150 | -0.493 | -10.7702 | -10.8024 | +32.2 | 59 | orbital |
| N2 | r2SCAN | +35.224 | +35.240 | 150 | -0.016 | -10.7261 | -10.7267 | +0.6 | 35 | orbital |
| N2 | TPSS | +8.420 | +8.392 | 150 | +0.028 | -10.4615 | -10.4621 | +0.6 | 27 | density |
| CO | SCAN (Max_Iter) | +22.765 | +22.822 | 150 | -0.057 | -9.5954 | -9.5976 | +2.2 | 200 | orbital |
| CO | r2SCAN | +19.591 | +19.736 | 150 | -0.145 | -9.5127 | -9.5134 | +0.7 | 44 | orbital |
| CO | TPSS | +1.575 | +1.589 | 150 | -0.014 | -9.2826 | -9.2832 | +0.6 | 43 | density |
| O2 | SCAN | +19.193 | +18.248 | 300 | +0.945 | -7.4962 / -11.5994 | -7.5098 / -11.6164 | +13.6 / +17.0 | 41 | orbital |
| O2 | r2SCAN | +13.137 | +13.471 | 300 | -0.334 | -7.4410 / -11.6889 | -7.4424 / -11.6907 | +1.4 / +1.8 | 28 | orbital |
| O2 | TPSS | +1.759 | +1.786 | 250 | -0.027 | -7.0612 / -11.5640 | -7.0621 / -11.5653 | +0.9 / +1.3 | 30 | density |

r2SCAN and TPSS agree within 0.15 mRy (O2 r2SCAN 0.33 mRy) and their HOMOs within 1.8 meV. SCAN agrees within
0.06 to 0.5 mRy (O2 0.95 mRy) and its HOMOs within 40 meV: SCAN is grid-sensitive in every code (QE's own SCAN
E - E_PBE moves by 0.1 to 0.7 mRy between 150 and 300 Ry, `qe/README.md`).

**Why a meta-GGA differs more than PBE at the same grid: quadrature.** The energy is `h^3 sum_i e(n_i, sigma_i,
tau_i)` over the grid points. `tau` and the iso-orbital indicator `alpha = (tau - tau_W)/tau_unif` vary on shorter
length scales than the density, so the sum carries a quadrature (aliasing) error that changes sign from one grid
spacing to the next; a plane-wave code has the same error on its dense FFT grid. The derivative stencil of `sigma`
and `tau`, the boundary of the domain and the form of the density gradient contribute at most 0.07 mRy: with the
energy density summed on a twice finer grid (the orbitals interpolated spectrally) r2SCAN and TPSS agree with QE to
0.04 mRy and 2.5 meV at h = 0.20, 0.16 and 0.13 bohr. SCAN's steep switching function in `alpha` makes its error the
largest. `python -m parsec_python.acceleration.benchmarks.meta_gga_accuracy --input <parsec.in> --refine 2` estimates
this error for an input ("Accuracy and grid spacing" in `src/parsec_python/README.md`).

## Grid spacing

r2SCAN at h = 0.13 bohr (`*_h013` inputs, E_PBE at the same h) against QE at 300 Ry:

| molecule | h (bohr) | E_r2SCAN - E_PBE this code (mRy) | QE 300 Ry (mRy) | this code - QE (mRy) | HOMO this code (eV) | dHOMO (meV) | SCF iterations |
|---|---|---|---|---|---|---|---|
| H2O | 0.16 | -2.132 | -2.024 | -0.109 | -7.5678 | +1.0 | 31 |
| H2O | 0.13 | -1.977 | -2.024 | +0.046 | -7.5676 | +1.2 | 43 |
| O2 | 0.16 | +13.137 | +13.471 | -0.334 | -7.4410 / -11.6889 | +1.4 / +1.8 | 28 |
| O2 | 0.13 | +13.577 | +13.471 | +0.106 | -7.4408 / -11.6897 | +1.6 / +1.0 | 35 |

The difference changes sign and shrinks (the quadrature error above). Advice:

- r2SCAN, TPSS and other smooth meta-GGAs: `Grid_Spacing` 0.16 bohr gives E - E_PBE within about 0.1 mRy (O2 0.35)
  and levels within 2 meV; 0.13 bohr for energy differences to 0.05-0.1 mRy with first-row atoms (1.9 times the grid
  points of 0.16, 2.3 times the time here). 0.20 bohr is too coarse for oxygen (O2 r2SCAN: 1 mRy, 14 meV).
- SCAN: 0.13 bohr or finer, and still up to 0.2 mRy and 25 meV; prefer r2SCAN (built to remove this sensitivity)
  unless SCAN itself is required.
- LUMOs of small molecules near 0 eV are diffuse and confined by `Boundary_Sphere_Radius`: not compared here.

## Meta-GGA hybrids against QE

QE at 150 Ry, `ecutfock` 300 Ry; E - E_PBE (mRy) with QE's PBE at 150 Ry; a_x E_x = the exact-exchange energy of the
final orbitals (TPSSh 0.10, r2SCAN0 0.25 of E_x^HF). QE 7.5 gives no O2 TPSSh or r2SCAN0 (`qe/README.md`).

| molecule | functional | E - E_PBE this code (mRy) | E - E_PBE QE (mRy) | this code - QE (mRy) | a_x E_x this code (Ry) | a_x E_x QE (Ry) | difference (mRy) | HOMO this code (eV) | HOMO QE (eV) | dHOMO (meV) | inner SCF per outer iteration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H2O | TPSSh | -8.113 | -8.186 | +0.073 | -0.78720937 | -0.78721403 | +0.005 | -8.0318 | -8.0323 | +0.5 | [6, 9, 14] |
| H2O | r2SCAN0 | +11.265 | +11.303 | -0.038 | -1.97346371 | -1.97347758 | +0.014 | -9.2892 | -9.2906 | +1.4 | [11, 12, 17, 8] |
| CH4 | TPSSh | -41.476 | -41.426 | -0.050 | -0.59343222 | -0.59342921 | -0.003 | -10.2559 | -10.2556 | -0.3 | [12, 10, 15, 9] |
| CH4 | r2SCAN0 | -3.799 | -3.745 | -0.054 | -1.48112791 | -1.48112872 | +0.001 | -11.2307 | -11.2306 | -0.1 | [7, 14, 9, 7] |
| NH3 | TPSSh | -16.874 | -16.873 | -0.001 | -0.67179881 | -0.67180198 | +0.003 | -6.8437 | -6.8434 | -0.3 | [7, 15, 11] |
| NH3 | r2SCAN0 | +11.177 | +11.158 | +0.019 | -1.68126314 | -1.68127045 | +0.007 | -7.9244 | -7.9234 | -1.0 | [12, 20, 16, 16] |
| N2 | TPSSh | +18.447 | +18.419 | +0.028 | -0.89752338 | -0.89752649 | +0.003 | -11.2062 | -11.2070 | +0.8 | [6, 6, 9] |
| N2 | r2SCAN0 | +61.904 | +61.921 | -0.018 | -2.24537516 | -2.24536486 | -0.010 | -12.4850 | -12.4856 | +0.6 | [8, 6, 17, 5] |
| CO | TPSSh | +9.782 | +9.795 | -0.013 | -0.93593418 | -0.93593890 | +0.005 | -9.9364 | -9.9371 | +0.7 | [6, 9, 9] |
| CO | r2SCAN0 | +43.956 | +44.056 | -0.100 | -2.34422107 | -2.34421579 | -0.005 | -11.0408 | -11.0414 | +0.6 | [6, 15, 13, 10, 9] |

E - E_PBE agrees within 0.1 mRy, the exact exchange within 0.014 mRy and the HOMOs within 1.4 meV: the meta-GGA part
at the level of the semilocal r2SCAN and TPSS above (QE's own 150 Ry rows of r2SCAN and TPSS are 0.005 to 0.12 mRy
from 300 Ry), the exact exchange at the level of the GGA hybrids of `../0d_hybrid_functionals`. The Fock loops take 3
or 4 outer iterations (CO r2SCAN0 5) to an estimated exchange error below 1e-8 Ry.

## Forces

The distorted water molecule (one O-H bond 1.10 A), forces after the net force is removed (Ry/bohr; y = 0 by
symmetry), against QE at 200 Ry; the last column is QE's own change from 150 to 200 Ry:

| functional | atom | this code (x, z) | QE 200 Ry (x, z) | this code - QE 200 Ry (x, z) | QE 150 - QE 200 Ry (x, z) |
|---|---|---|---|---|---|
| PBE | O | +0.151519, +0.054879 | +0.151003, +0.055011 | +5.2e-04, -1.3e-04 | -5.3e-05, -4.8e-05 |
| PBE | H1 (1.10 A) | -0.122967, -0.086557 | -0.123481, -0.086608 | +5.1e-04, +5.1e-05 | +3.3e-05, +2.7e-05 |
| PBE | H2 | -0.028552, +0.031678 | -0.027522, +0.031597 | -1.0e-03, +8.1e-05 | +2.0e-05, +2.2e-05 |
| r2SCAN | O | +0.149123, +0.071822 | +0.149053, +0.073281 | +7.0e-05, -1.5e-03 | -2.0e-04, -2.5e-04 |
| r2SCAN | H1 (1.10 A) | -0.136115, -0.094392 | -0.136854, -0.095122 | +7.4e-04, +7.3e-04 | +1.2e-04, +1.2e-04 |
| r2SCAN | H2 | -0.013008, +0.022570 | -0.012198, +0.021841 | -8.1e-04, +7.3e-04 | +7.8e-05, +1.3e-04 |

The largest difference is 1.0e-3 Ry/bohr for PBE and 1.5e-3 for r2SCAN at h = 0.16 bohr (at 0.13: 2.6e-4 and
5e-4). The meta-GGA forces are the derivatives of the grid energy with the force formulas of the GGAs (the orbital
term does not depend on the atoms; finite differences of the energy match them as for PBE), so their difference to
QE is the grid error of the energy, larger for a meta-GGA as above.

## SCAN

- **Residual floor.** The SCAN runs of CH4, NH3 and CO stop at `Max_Iter` 200 ("Maximum SCF iterations reached
  without convergence") with the total energy settled to the printed 1e-8 Ry from iteration 24, 48 and 26 on, while
  the weighted residual stays between 3e-6 and 1.4e-4: points of density 1e-5 to 1e-3, where SCAN's switching
  function in `alpha` amplifies the last digits of `tau - tau_W`. r2SCAN, built to remove that sensitivity, converges
  as a GGA does. For SCAN energies, `Convergence_Criterion: 1e-4 Ry` stops these runs after 30 to 52 iterations
  within 6e-8 Ry of the energies above.
- **Fine grids.** The density-weighted Anderson mixer lets the low-density values of `vtau` and `W` drift; on fine
  grids that can produce states localized on a few points (SCAN H2O at h = 0.13 bohr: passing excursions, residual
  4.6e-5 after 80 iterations). `PARSEC_META_GGA_ANDERSON=plain` converges it (54 iterations), as does r2SCAN.

## Costs

`main.py` on one A100: program wall time (setup included, one process per input) / SCF iterations (hybrids: the PBE
start + the inner iterations of the Fock loop) / peak device memory of the process. Grid: 1.4 to 1.8 million points
at h = 0.16 bohr (H2O 1,568,208), 3 million at 0.13.

| input | PBE | SCAN | r2SCAN | TPSS | TPSSh | r2SCAN0 |
|---|---|---|---|---|---|---|
| H2O, h 0.16 | 9 s / 44 / 1.2 GiB | 31 s / 45 / 3.0 GiB | 25 s / 31 / 2.7 GiB | 24 s / 43 / 2.5 GiB | 29 s / 20 + 29 / 3.0 GiB | 49 s / 20 + 48 / 3.3 GiB |
| CH4, h 0.16 | 9 s / 36 / 1.5 GiB | 108 s / 200 (Max_Iter) / 3.6 GiB | 29 s / 29 / 3.2 GiB | 26 s / 28 / 3.0 GiB | 33 s / 13 + 46 / 3.5 GiB | 47 s / 13 + 37 / 3.8 GiB |
| NH3, h 0.16 | 14 s / 41 / 2.2 GiB | 149 s / 200 (Max_Iter) / 4.5 GiB | 59 s / 62 / 3.7 GiB | 37 s / 45 / 3.7 GiB | 45 s / 19 + 33 / 4.2 GiB | 79 s / 19 + 64 / 4.4 GiB |
| N2, h 0.16 | 6 s / 25 / 0.9 GiB | 28 s / 59 / 2.4 GiB | 22 s / 35 / 2.3 GiB | 19 s / 27 / 2.1 GiB | 22 s / 14 + 21 / 2.6 GiB | 33 s / 14 + 36 / 2.8 GiB |
| CO, h 0.16 | 9 s / 32 / 1.1 GiB | 92 s / 200 (Max_Iter) / 2.8 GiB | 28 s / 44 / 2.5 GiB | 23 s / 43 / 2.3 GiB | 29 s / 27 + 24 / 2.8 GiB | 44 s / 27 + 53 / 3.0 GiB |
| O2, h 0.16 | 12 s / 29 / 1.1 GiB | 39 s / 41 / 3.5 GiB | 32 s / 28 / 2.5 GiB | 30 s / 30 / 2.3 GiB |  |  |
| H2O, h 0.13 | 16 s / 44 / 1.8 GiB |  | 59 s / 43 / 4.7 GiB |  |  |  |
| O2, h 0.13 | 23 s / 37 / 1.6 GiB |  | 72 s / 35 / 4.2 GiB |  |  |  |
| H2O distorted (forces), h 0.16 | 11 s / 28 / 2.1 GiB |  | 32 s / 27 / 3.5 GiB |  |  |  |

A meta-GGA run takes 2.5 to 4.2 times the time of PBE here (the SCAN runs stopped at `Max_Iter` more). An SCF iteration costs 2 to 3 times as much (`tau` of the
orbitals, the orbital term `-div(vtau grad)` in every Hamiltonian application and the wider Chebyshev interval it
brings: H2O 0.17 s per iteration with PBE, 0.33 s with TPSS, 0.46-0.51 s with SCAN and r2SCAN), and the set-up
before the SCF ("Pre-SCF setup/reporting wall time" of the log) takes 8 to 12 s for a meta-GGA against 1.3 to 1.7 s
for PBE (17 to 18 s at h = 0.13 bohr). The meta-GGA hybrids add 3 to 24 s to the semilocal meta-GGA run.
