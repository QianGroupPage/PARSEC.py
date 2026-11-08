# Quantum ESPRESSO side

Inputs `<molecule>_<functional>_e<ecutwfc>.in` of the reference values in `../README.md` (QE 7.5 with libxc 7.0.0,
pw.x; NERSC's module `espresso/7.5-libxc-7.0.0-cpu` and a conda-forge build give the same totals to 1e-8 Ry). Run
with the UPF files in `./pseudo` (below), e.g. `mpirun -np 16 pw.x -in H2O_r2scan_e300.in`. The numbers are in
`QE` of `../collect_results.py`.

## Pseudopotentials

The UPF files are the FHI98PP tables of `src/pp_generation/NCPPs/PBE/<E>/FHI/` (`../<E>_FHIPP.DAT`) in
Kleinman-Bylander form with the same local channel (d for C, N, O; s for H) and one projector per l (QE's own FHI
reader). The repository's `src/pp_generation/NCPPs/PBE/<E>/UPF/<e>_pbe_tm.UPF` hold the same tables; copy them into
`qe/pseudo/`. They have no nonlinear core correction, so the question of a core density in `tau` does not arise.
They are PBE potentials, as is usual for meta-GGA calculations (few meta-GGA-generated potentials exist); both codes
use the same ionic potential, so this inconsistency is the same in both and cancels in every comparison.

## Settings

- Geometries (bohr) of the parsec inputs, centred in a cubic box of L = 20 bohr, `assume_isolated = 'mt'`, Gamma
  only, `occupations = 'fixed'`, `nbnd` = occupied + 4, `ecutrho = 4 ecutwfc`, `mixing_beta = 0.4`,
  `conv_thr = 1e-9` Ry (hybrids and the distorted water 1e-10). O2: `nspin = 2`, `tot_magnetization = 2`.
  L 20 against 24 bohr: within 1.4e-6 Ry and 0.2 meV (H2O: PBE, SCAN, r2SCAN).
- Functionals through libxc: SCAN `XC-000I-000I-000I-000I-263L-267L`, r2SCAN `...-497L-498L`, TPSS `...-202L-231L`;
  the hybrids TPSSh `XC-000I-000I-000I-000I-000I-457L` and r2SCAN0 `...-660L` (an exchange-correlation hybrid of
  libxc in QE's meta-GGA slot: QE takes the exact-exchange fraction from libxc and does not rescale it) with
  `exxdiv_treatment = 'vcut_spherical'`, `x_gamma_extrapolation = .false.`, `nqx = 1 1 1`, `ecutfock = 300` Ry,
  converged to dexx < 1e-10 Ry. E - E_PBE always takes QE's PBE of the same box and cutoff.
- QE 7.5 switches off libxc 7's Fermi-hole-curvature clamp for every libxc functional (`xclib_init_libxc`); with it
  on, the converged SCAN, r2SCAN and M06-L energies are the same to 1e-8 Ry (plane waves keep
  `sigma <= 8 n tau` to round-off). QE drops grid points below `rho` 1e-12 and `tau` 1e-12 (1e-10 per spin channel).
- QE 7.5 gives no O2 TPSSh or r2SCAN0: both diverge in the SCF without exact exchange that QE runs first (for an
  exchange-correlation hybrid of libxc that part already lacks `a_x E_x`), with every mixing tried.

## Cutoff

E - E_PBE (mRy) of QE at the cutoffs of the convergence rows (L 20, dual 4); `../README.md` compares with the 300 Ry
column (TPSS of H2O and O2: 250 Ry), the other rows with 150 Ry:

| molecule | functional | 150 Ry | 200 Ry | 250 Ry | 300 Ry |
|---|---|---|---|---|---|
| H2O | r2SCAN | -2.038 | -2.018 | -2.017 | -2.024 |
| H2O | SCAN | +1.543 | +1.857 | +1.800 | +1.762 |
| H2O | TPSS | -11.145 | - | -11.083 | - |
| CH4 | r2SCAN | -1.422 | -1.443 | -1.440 | -1.443 |
| CH4 | SCAN | -3.821 | -3.889 | -3.897 | -3.924 |
| NH3 | r2SCAN | +4.828 | +4.829 | +4.818 | +4.822 |
| NH3 | SCAN | +6.491 | +6.577 | +6.537 | +6.516 |
| O2 | r2SCAN | +13.351 | +13.504 | +13.443 | +13.471 |
| O2 | SCAN | +18.074 | +18.749 | +18.290 | +18.248 |
| O2 | TPSS | +1.681 | - | +1.786 | - |

r2SCAN and TPSS settle to about 0.01 mRy above 200 Ry (O2 0.06 mRy); at 150 Ry they are 0.005 to 0.12 mRy off
(the error of the 150 Ry rows of N2, CO and the hybrids is of that size). SCAN scatters by 0.1 (H2O) to 0.5 mRy (O2)
between 200 and 300 Ry without settling: its steep switching function in `alpha` makes the quadrature of its energy
density grid-sensitive in a plane-wave code as well (a shift of the molecule by 0.0625 bohr moves QE's SCAN HOMO of
H2O by 27 meV, r2SCAN's by 0.9 meV). HOMOs of r2SCAN and TPSS are stable to 1 meV from 150 Ry; SCAN's to 0.01-0.02 eV.
Forces of the distorted water molecule, 150 against 200 Ry: PBE 5e-5, r2SCAN 2.5e-4 Ry/bohr.
