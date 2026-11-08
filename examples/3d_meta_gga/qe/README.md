# Quantum ESPRESSO side

Inputs `Si_<functional>_k<mesh>_e200.in` of the reference values in `../README.md` (QE 7.5 with libxc 7.0.0, pw.x;
the numbers are in `QE` of `../collect_results.py`). Run with the UPF file in `./pseudo`, e.g.
`mpirun -np 16 pw.x -nk 4 -in Si_r2scan_k444_e200.in`.

- Pseudopotential: the FHI98PP table of `../Si_FHIPP.DAT` in Kleinman-Bylander form (local d, one projector per l,
  QE's own FHI reader); `src/pp_generation/NCPPs/PBE/Si/UPF/si_pbe_tm.UPF` holds the same table (copy it into
  `qe/pseudo/`). No core correction. QE integrates the Kleinman-Bylander energies by Simpson's rule, this code (as
  PARSEC) by the trapezoid on the logarithmic mesh: QE with this code's coefficients lowers its PBE energy by 3.3e-4
  to 3.6e-4 Ry per cell, the PBE offset of `../README.md`, which cancels in E - E_PBE.
- fcc cell `ibrav = 0` with the lattice vectors of the parsec inputs (a = 10.26 bohr), atoms at 0 and a/4 (1,1,1);
  `K_POINTS automatic n n n 0 0 0` (Gamma-centred, reduced by QE's symmetry); `ecutwfc = 200` Ry, `ecutrho = 1600` Ry
  (dual 8); `occupations = 'fixed'`, `nbnd = 8`, `conv_thr = 1e-10` Ry; SCAN `XC-000I-000I-000I-000I-263L-267L`,
  r2SCAN `...-497L-498L`, TPSS `...-202L-231L` (libxc's FHC clamp switched off by QE, which changes nothing at
  convergence here).
- Cutoff: PBE at 200 Ry is within 1e-7 Ry per cell of 250 Ry. The meta-GGA totals scatter by about +-1.5e-4 Ry per
  cell for every cutoff from 120 to 250 Ry and either dual (the quadrature of the meta-GGA energy density on the dense
  FFT grid), the gaps by 1-2 meV.
- The meshes are references for a comparison at equal sampling, not converged Brillouin-zone integrals.
