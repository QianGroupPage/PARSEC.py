# Quantum ESPRESSO side

Inputs `<system>_<functional>_k<mesh>_e<ecutwfc>[_<q+G=0>].in` of the reference values in `../README.md` (QE 7.5 with
libxc 7.0.0, pw.x; the conda-forge build and NERSC's module `espresso/7.5-libxc-7.0.0-cpu` give the same totals to
1e-8 Ry), and `*_moved.in`, the same inputs with every atom moved off QE's FFT grid (`../README.md`, "QE's HSE energies
and its grid"). Run with the UPF files in `./pseudo` (below), e.g.
`mpirun -np 16 pw.x -nk 4 -in si_pbe0_k444_e60_gbx.in`.

## Pseudopotentials

The UPF files are the FHI98PP tables of `src/pp_generation/NCPPs/PBE/<E>/FHI/` (`../<E>_FHIPP.DAT`) in
Kleinman-Bylander form with the same local channel (d) and one projector per l, converted by QE's own FHI reader
(`fhi2upf.x`); Mg and B carry their nonlinear core correction. The repository's
`src/pp_generation/NCPPs/PBE/<E>/UPF/<e>_pbe_tm.UPF` hold the same tables (they differ in header attributes and the
label of the unused d wavefunction); copy them into `qe/pseudo/`. The local potentials, projectors, pseudo-wavefunctions
and core charges of the UPF files equal the FHI tables as this code reads them to 1e-15. The Kleinman-Bylander
energies E_l differ by -0.9e-4 to -1.1e-4 relative (QE integrates them by Simpson's rule, this code, as PARSEC, by the
trapezoid on the logarithmic mesh): a constant offset of the totals, which mostly cancels in E_hyb - E_PBE.

## Settings

- Cells and atoms (bohr) of the parsec inputs; h-BN with a cell height of 30 bohr (= twice the slab half height of
  the parsec input) and `assume_isolated = '2D'`, which cuts the Hartree, local and Ewald terms; QE 7.5 does not cut
  the exact exchange (its Fock term is that of the stack of layers 30 bohr apart).
- `K_POINTS automatic n n n 0 0 0` (Gamma-centred), reduced by symmetry for the orbitals; the exchange runs over the
  full mesh, `nqx1..3 = n` (the q-mesh equals the k-mesh).
- `ecutwfc` Si 60, C 120, MgO 150, h-BN 100 (and 160, `_e160`), O2 150 Ry; `ecutfock = 2 ecutwfc` (equal to 4 ecutwfc
  to 1e-8 Ry); `conv_thr = 1e-10` Ry, which also stops the exact-exchange loop (dexx <= 1e-10 Ry at the end);
  `occupations = 'fixed'`; `nbnd` = occupied + 4; ACE (QE's default). O2: `nspin = 2`, `tot_magnetization = 2`.
- Functionals: `pbe0`; `hse` with `screening_parameter = 0.11` (= this code's `hse06`: one omega in the exact and the
  semilocal part; QE's default is 0.106); `hf`; B3LYP through libxc 402 (`XC-000I-000I-402L-000I-000I-000I`, VWN-RPA,
  this code's `b3lyp`).
- q+G = 0 (`exxdiv_treatment = 'gygi-baldereschi'`): `_gbx` with `x_gamma_extrapolation = .true.` (QE's default), `_gb`
  with `.false.`; the auxiliary function of QE has the exponent 10/ecutwfc (bohr^2).
- `_moved`: every atom shifted by a quarter of a step of QE's dense FFT grid along a1 (Si 36, C 36, MgO 45 points
  along a1; h-BN 32 points along a1 = (a, 0, 0)) and, for O2, by half a step (0.0625 bohr) along x, y and z, so that
  no point where the gradient of the density vanishes by symmetry (atom sites, bond centres, the O2 bond midpoint) is a
  point of QE's grid.

## QE numbers

E in Ry; E - E_PBE in Ry against the `pbe` input of the same system, mesh and cutoff (the 160 Ry rows against PBE at
160 Ry); Fock = `+ Fock energy (ACE)` = a_x E_x^HF in Ry; gap = CBM - VBM over the k-points of the mesh in eV (O2:
lowest empty minus highest occupied level of both channels); VBM in eV in QE's energy zero; outer =
exact-exchange iterations; the q+G = 0 element is QE's kernel value at q + G = 0 (`-exxdiv`).

| input | E (Ry) | E - E_PBE (Ry) | Fock (Ry) | gap (eV) | VBM (eV) | k-points (irreducible) | outer | q+G=0 element (Ry bohr^3) |
|---|---|---|---|---|---|---|---|---|
| si_pbe_k222_e60 | -15.57477255 |  |  | 0.6105 | 6.5274 | 3 |  |  |
| si_pbe0_k222_e60_gbx | -15.59118868 | -0.01641613 | -1.09610497 | 2.2893 | 5.5554 | 3 | 6 | 827.3784 |
| si_hse06_k222_e60_gb | -15.59495620 | -0.02018365 | -0.86447055 | 1.6231 | 5.8914 | 3 | 6 | 429.2121 |
| si_hf_k222_e60_gbx | -15.07439749 | +0.50037506 | -4.46160434 | 7.9362 | 3.6183 | 3 | 14 | 827.3784 |
| si_b3lyp_libxc_k222_e60_gbx | -15.53650910 | +0.03826345 | -0.88257822 | 2.2131 | 5.8769 | 3 | 8 | 827.3784 |
| si_pbe_k444_e60 | -15.74854796 |  |  | 0.7006 | 6.2737 | 8 |  |  |
| si_pbe0_k444_e60_gbx | -15.75913239 | -0.01058443 | -1.07070707 | 1.9817 | 5.5548 | 8 | 6 | 3309.5138 |
| si_hse06_k444_e60_gb | -15.75599559 | -0.00744763 | -0.83294089 | 1.3274 | 5.8841 | 8 | 6 | 519.1727 |
| si_hf_k444_e60_gbx | -15.21710704 | +0.53144092 | -4.36746762 | 6.7861 | 4.2046 | 8 | 15 | 3309.5138 |
| si_disp_pbe_k222_e60 | -15.57432061 |  |  | 0.4113 | 6.6589 | 5 |  |  |
| si_disp_pbe0_k222_e60_gbx | -15.59071369 | -0.01639308 | -1.09607990 | 2.0801 | 5.6983 | 5 | 6 | 827.3784 |
| si_disp_hse06_k222_e60_gb | -15.59448117 | -0.02016056 | -0.86444827 | 1.4149 | 6.0333 | 5 | 5 | 429.2121 |
| c_pbe_k333_e120 | -22.75901823 |  |  | 4.1749 | 13.3591 | 4 |  |  |
| c_pbe0_k333_e120_gbx | -22.76126647 | -0.00224824 | -1.60903339 | 6.2152 | 12.2822 | 4 | 5 | 803.3629 |
| c_hse06_k333_e120_gb | -22.76345019 | -0.00443196 | -1.36981661 | 5.5510 | 12.5965 | 4 | 4 | 424.9761 |
| mgo_pbe_k333_e150 | -35.13885358 |  |  | 4.6579 | 5.6964 | 4 |  |  |
| mgo_pbe0_k333_e150_gbx | -34.86879976 | +0.27005382 | -1.89795571 | 7.2118 | 4.0803 | 4 | 6 | 1120.5162 |
| mgo_hse06_k333_e150_gb | -34.91008875 | +0.22876483 | -1.65476926 | 6.4986 | 4.4479 | 4 | 6 | 476.6566 |
| hbn_pbe_k661_e100 | -25.80737855 |  |  | 4.6973 | -5.8270 | 7 |  |  |
| hbn_pbe0_k661_e100_gbx | -25.79687397 | +0.01050458 | -1.68611913 | 6.3753 | -6.9732 | 7 | 6 | 3702.9928 |
| hbn_hse06_k661_e100_gb | -25.79627181 | +0.01110674 | -1.44206026 | 5.7089 | -6.5915 | 7 | 5 | 519.1686 |
| hbn_pbe_k661_e160_e160 | -25.80788400 |  |  | 4.6979 | -5.8273 | 7 |  |  |
| hbn_pbe0_k661_e160_gb_e160 | -25.79825322 | +0.00963078 | -1.68700769 | 6.3935 | -6.9896 | 7 | 6 |  |
| hbn_hse06_k661_e160_gb_e160 | -25.79668246 | +0.01120154 | -1.44208798 | 5.7107 | -6.5918 | 7 | 5 |  |
| o2_pbe_k111_e150 | -63.71072842 |  |  | 2.4864 | -6.1135 | 1 |  |  |
| o2_pbe0_k111_e150_gbx | -63.67028508 | +0.04044334 | -3.16297699 | 6.1969 | -8.2690 | 1 | 6 | 700.4071 |
| o2_hse06_k111_e150_gb | -63.67360858 | +0.03711984 | -2.80020128 | 5.4187 | -7.8820 | 1 | 6 | 389.0709 |

Forces of the displaced Si cell (atom 2, Ry/bohr; atom 1 carries the opposite force): PBE (-0.01806913, 0, 0), PBE0
gbx (-0.01898746, 0, 0), HSE06 gb (-0.01889558, 0, 0).

### Atoms moved off QE's grid (`*_moved.in`)

Run with 32 or 64 MPI ranks of one CPU node (the same QE 7.5 and libxc 7.0.0). E - E_PBE against the moved
`pbe` input; change = this E - E_PBE minus that of the reference row above.

| input | E (Ry) | E - E_PBE (Ry) | change (Ry) | Fock (Ry) |
|---|---|---|---|---|
| si_pbe_k222_e60_moved | -15.57477219 |  |  |  |
| si_pbe0_k222_e60_gbx_moved | -15.59118805 | -0.01641586 | +2.7e-07 | -1.09610496 |
| si_hse06_k222_e60_gb_moved | -15.59495330 | -0.02018111 | +2.5e-06 | -0.86447081 |
| si_pbe_k444_e60_moved | -15.74854757 |  |  |  |
| si_pbe0_k444_e60_gbx_moved | -15.75913194 | -0.01058437 | +6.0e-08 | -1.07070698 |
| si_hse06_k444_e60_gb_moved | -15.75599151 | -0.00744394 | +3.7e-06 | -0.83294150 |
| c_pbe_k333_e120_moved | -22.75901773 |  |  |  |
| c_pbe0_k333_e120_gbx_moved | -22.76126429 | -0.00224656 | +1.7e-06 | -1.60903333 |
| c_hse06_k333_e120_gb_moved | -22.76342768 | -0.00440995 | +2.2e-05 | -1.36981662 |
| mgo_pbe_k333_e150_moved | -35.13885377 |  |  |  |
| mgo_pbe0_k333_e150_gbx_moved | -34.86880028 | +0.27005349 | -3.3e-07 | -1.89795566 |
| mgo_hse06_k333_e150_gb_moved | -34.91003581 | +0.22881796 | +5.3e-05 | -1.65476929 |
| hbn_pbe_k661_e100_moved | -25.80737613 |  |  |  |
| hbn_pbe0_k661_e100_gbx_moved | -25.79687242 | +0.01050371 | -8.7e-07 | -1.68611887 |
| hbn_hse06_k661_e100_gb_moved | -25.79626094 | +0.01111519 | +8.5e-06 | -1.44206011 |
| o2_pbe_k111_e150_moved | -63.71074166 |  |  |  |
| o2_pbe0_k111_e150_gbx_moved | -63.67028562 | +0.04045604 | +1.3e-05 | -3.16297971 |
| o2_hse06_k111_e150_gb_moved | -63.67330643 | +0.03743523 | +3.2e-04 | -2.80019512 |

The Fock energies do not move (at most 6e-6 Ry, O2 HSE06: the orbitals of a slightly different semilocal potential);
the PBE totals move by 2e-7 (MgO) to 1.3e-5 Ry (O2), the PBE0 differences E - E_PBE by at most 1.3e-5 Ry (O2), and
the HSE06 differences by +2.5e-6 (Si 2x2x2) to +3.2e-4 Ry (O2).
