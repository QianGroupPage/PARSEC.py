# Meta-GGAs of bulk Si against Quantum ESPRESSO

SCAN (this code's native SCAN), r2SCAN and TPSS (libxc) of bulk Si, fcc with two atoms, on Gamma-centred 2x2x2 and
4x4x4 meshes, compared with Quantum ESPRESSO 7.5 + libxc 7.0.0 using the same FHI98PP potential and cell. The
quantities compared are E_mGGA - E_PBE of each code at its own discretization (per cell) and the band gap on the
mesh. The molecules, the meta-GGA hybrids, the keywords and the defaults are in
[`../0d_meta_gga/README.md`](../0d_meta_gga/README.md).

## Contents

- `Si_k222/<functional>/parsec.in`, `Si_k444/<functional>/parsec.in` for `pbe`, `scan`, `r2scan`, `tpss`, and
  `parsec_python.out`, the reference log of that input (one NVIDIA A100; paths shortened to repository paths, the libxc
  library as `$PARSEC_LIBXC_LIBRARY`).
- `Si_FHIPP.DAT`: a copy of `src/pp_generation/NCPPs/PBE/Si/FHI/Si_FHIPP.DAT` (Troullier-Martins, PBE, no core
  correction), found with `--pp-dir`.
- `qe/`: the Quantum ESPRESSO inputs, their settings and the cutoff error (`qe/README.md`).
- `run_examples.py` (one process per input, `main.py --gpus 1`; `--install` copies the logs of a run folder into the
  example) and `collect_results.py` (the tables below from the logs).

## Run

From the repository root, with libxc for r2SCAN and TPSS (see "Functionals of libxc" in
`src/parsec_python/README.md`):

```bash
export PARSEC_LIBXC_LIBRARY=.../libxc-7.1.2-cuda-sm80/lib/libxc.so
python src/parsec_python/main.py examples/3d_meta_gga/Si_k444/r2scan/parsec.in --pp-dir examples/3d_meta_gga --no-archive
python examples/3d_meta_gga/run_examples.py --out RUNS          # every input, logs in RUNS/<mesh>/<functional>/
python examples/3d_meta_gga/collect_results.py [--runs RUNS]
```

Settings of every input: a = 10.26 bohr (lattice vectors (0, a/2, a/2), ...), atoms at 0 and a/4 (1, 1, 1),
`Grid_Spacing` 0.20 bohr, `Expansion_Order` 12, `Cubic_Spline`, `States_Num` 8, `Fermi_Temp` 10 K (occupations 1
and 0), `Convergence_Criterion` 1e-6 Ry. The mesh is `Monkhorst_Pack_Grid n n n` with a zero
`Monkhorst_Pack_Shift`: QE's Gamma-centred mesh, reduced to its irreducible wedge (3 and 8 k-points) with the
density symmetrized; these cells run on the Fortran-derived routines (the default for these grids, as the log says).
The functional lines are those of the molecules: `Correlation_Type: scan`, `LibXC_Functional: mgga_x_r2scan
mgga_c_r2scan`, and TPSS in PARSEC 1.5's form (`Use_LibXC`, `Func_Id_X: 202`, `Func_Id_C: 231`). The default
density gradient (`auto`) is the orbital form for SCAN and r2SCAN and the density form for TPSS (its log says so).

## Against Quantum ESPRESSO

QE at 200 Ry, dual 8. PBE:

| mesh | E this code (Ry/cell) | E QE (Ry/cell) | difference (Ry) | gap this code (eV) | gap QE (eV) | dgap (meV) |
|---|---|---|---|---|---|---|
| 2x2x2 | -15.57525517 | -15.57491928 | -3.36e-04 | 0.6102 | 0.6102 | +0.0 |
| 4x4x4 | -15.74905936 | -15.74871754 | -3.42e-04 | 0.7003 | 0.7002 | +0.1 |

The PBE offset is PARSEC's Kleinman-Bylander quadrature (trapezoid on the logarithmic mesh against QE's Simpson
rule; QE with this code's coefficients moves by -3.3e-4 to -3.6e-4 Ry per cell, `qe/README.md`); it cancels in
E - E_PBE. Meta-GGAs, E - E_PBE in mRy per cell:

| functional | mesh | E - E_PBE this code (mRy) | E - E_PBE QE (mRy) | this code - QE (mRy) | gap this code (eV) | gap QE (eV) | dgap (meV) | SCF iterations | gradient form |
|---|---|---|---|---|---|---|---|---|---|
| SCAN | 2x2x2 | -7.777 | -7.687 | -0.089 | 0.7954 | 0.7932 | +2.2 | 16 | orbital |
| SCAN | 4x4x4 | -9.588 | -9.443 | -0.145 | 0.9654 | 0.9605 | +4.9 | 17 | orbital |
| r2SCAN | 2x2x2 | +0.407 | +0.516 | -0.109 | 0.7651 | 0.7648 | +0.3 | 13 | orbital |
| r2SCAN | 4x4x4 | -5.981 | -5.888 | -0.093 | 0.8564 | 0.8560 | +0.4 | 16 | orbital |
| TPSS | 2x2x2 | +14.551 | +14.650 | -0.099 | 0.9632 | 0.9628 | +0.4 | 23 | density |
| TPSS | 4x4x4 | +20.199 | +20.288 | -0.089 | 0.8712 | 0.8709 | +0.3 | 18 | density |

E - E_PBE agrees within 0.09 to 0.15 mRy per cell, which is QE's own scatter of the meta-GGA totals between 120 and
250 Ry (+-0.15 mRy per cell, `qe/README.md`); the gaps of r2SCAN and TPSS within 0.4 meV, SCAN's within 2 to 5 meV
(SCAN's grid sensitivity, see `../0d_meta_gga/README.md`; QE's SCAN gap moves by 1-2 meV with its cutoff). At
h = 0.16 bohr the differences are the same to 0.07 mRy (r2SCAN -0.10 and -0.09, SCAN -0.16 and -0.11 mRy).

## TPSS at the Gamma point alone

In a cell sampled at the Gamma point alone, every occupied orbital can have a zero gradient at the same grid points
(in Si the bond centres, at the density maximum), where `tau` and `sigma` vanish together. TPSS-type functionals
(`z = tau_W/tau`; the density form of the gradient under `auto`) may then stall or diverge: Si at Gamma stalls at an
SRE of 7e-3 at h = 0.25 bohr and diverges at h = 0.4, whatever `LibXC_Tau_Threshold` and `LibXC_Enforce_FHC` say.
Use a k-mesh (the 2x2x2 and 4x4x4 rows above converge in 18-23 iterations), or the orbital form
(`PARSEC_META_GGA_GRADIENT=orbital`: 125 and 173 iterations at h = 0.25 and 0.16 bohr, E - E_PBE within 0.05 and
0.25 mRy of QE, the gap within 26 and 7 meV). SCAN and r2SCAN, written with `alpha`, are not affected.

## Costs

`main.py` on one A100 (program wall time, setup included, one process per input; peak device memory of the process):

| mesh | irreducible k-points | functional | SCF iterations | wall (s) | peak device memory (GiB) |
|---|---|---|---|---|---|
| 2x2x2 | 3 | PBE | 13 | 13.3 | 0.7 |
| 2x2x2 | 3 | SCAN | 16 | 22.1 | 0.8 |
| 2x2x2 | 3 | r2SCAN | 13 | 19.4 | 0.8 |
| 2x2x2 | 3 | TPSS | 23 | 20.7 | 0.8 |
| 4x4x4 | 8 | PBE | 12 | 16.2 | 0.8 |
| 4x4x4 | 8 | SCAN | 17 | 29.6 | 1.0 |
| 4x4x4 | 8 | r2SCAN | 16 | 27.5 | 1.0 |
| 4x4x4 | 8 | TPSS | 18 | 28.3 | 1.0 |

The orbital terms of a meta-GGA (`tau` of the orbitals of every k-point and `-div(vtau grad)` in every Bloch
Hamiltonian) and the evaluation of libxc on the GPU make a meta-GGA run 1.5 to 1.8 times as long as PBE here.

## The check script

`python -m parsec_python.acceleration.benchmarks.meta_gga_periodic_check` writes the same cell with a functional,
grid spacing and mesh (also Gamma alone and the full mesh as an explicit k-point list), runs the k-point loop in its
own process (on the GPUs where CuPy finds them) and prints energy, band edges, gap, iterations and timings as JSON:

```bash
python -m parsec_python.acceleration.benchmarks.meta_gga_periodic_check --functional pbe,scan,r2scan,tpss \
    --kmesh 1,2,4 --spacing 0.20 [--gradient density|orbital] [--term-timing] [--program] --out results
```

`r2scan` and `tpss` are libxc's there (`LibXC_Functional`); `scan` is the native SCAN.
