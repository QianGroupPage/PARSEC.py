# CH₄ / CF₄ C 1s ΔSCF — PBE and B3LYP

Reproduces the CH₄ and CF₄ rows of Table S1 of our JCTC paper, and compares the chemical
shift ΔBE = BE(CF₄) − BE(CH₄) with experiment.

> Xu, Q.; Prendergast, D.; Qian, J. *Real-Space Pseudopotential Method for the Calculation
> of 1s Core-Level Binding Energies.* J. Chem. Theory Comput. **2022**, 18, 5471–5478.

## Running

```bash
export ARES_DSCF_DATA=/path/to/DSCF_DATA/C   # the JCTC 2022 runs (see Gotchas)
export ARES_PP_DIR=/path/to/PPs_copy         # their pseudopotentials
python3 setup.py            # build inputs (idempotent)
sbatch run8.qsub            # all 8 runs, one per node, debug queue
sbatch run_b3lyp_cf4.qsub   # b3lyp/CF4 only, if run8 times out on it
python3 analyze.py          # results
```

`analyze.py` skips runs that have not reached `Well Done`, so it is safe to call early.

## What is computed

ΔSCF in real-space KS-DFT with Dirichlet BCs. The core hole is fully screened, so each
state is an ordinary SCF run and `BE = AE_energy(final) − AE_energy(initial)`.

Use the **all-electron** energy, not the pseudo total energy: initial and final states use
pseudopotentials with different `z_valence`, so their pseudo energies are not on a common
scale. Absolute BEs then come out ≈6 eV high, because ARES evaluates eq 4 with
spin-unpolarized atomic energies. That offset is a per-element constant and **cancels in
ΔBE**, which is why ΔBE is the number compared to experiment.

Two trees, differing by one flag:

| tree     | `Lrefine` | reports                                             |
|----------|-----------|-----------------------------------------------------|
| `pbe/`   | `F`       | `*AE energy (test)` → PP-PBE                        |
| `b3lyp/` | `T`       | that **plus** `*ref.AE energy` → PP-PBE(B3LYP)      |

The B3LYP refinement is one non-self-consistent evaluation on the converged PBE density —
not a second SCF — so `Lrefine=T` alone yields both columns and is all new work needs.
`pbe/` is a control; `analyze.py` verifies the two agree on PP-PBE.

## Results

All 8 runs reached `Well Done` on 2026-08-14. Reference values recomputed from our JCTC
2022 logs, not transcribed from the SI.

Binding energies, eV (BE = AE[FS_1s] − AE[IS]; the ~+6 eV absolute offset is the
spin-unpolarized-atom artifact described above and cancels in ΔBE):

| run        | method        | BE(CH₄) | BE(CF₄) |        ΔBE |
|------------|---------------|--------:|--------:|-----------:|
| this work  | PP-PBE        | 297.148 | 307.715 |     10.568 |
| this work  | PP-PBE(B3LYP) | 297.233 | 308.425 |     11.192 |
| JCTC 2022  | PP-PBE        | 297.146 | 307.715 |     10.569 |
| JCTC 2022  | PP-PBE(B3LYP) | 297.232 | 308.425 |     11.193 |
| experiment | —             |  290.80 |  301.85 |  **11.05** |

PP-PBE is one result, not two: both trees compute it (`Lrefine=T` runs the same PBE SCF
before refining), and they agree to 3.3e-05 eV — that agreement is the control `pbe/`
exists for, so the column is listed once.

Reproduction of JCTC 2022 (this work − paper), eV:

| method        | ΔBE(CH₄) | ΔBE(CF₄) |  Δ(ΔBE) |
|---------------|---------:|---------:|--------:|
| PP-PBE        |  +0.0011 |  +0.0001 | −0.0011 |
| PP-PBE(B3LYP) |  +0.0010 |  −0.0000 | −0.0010 |

Reproduced to ~1 meV on ΔBE despite the different grid (see Gotchas).

Chemical shift against experiment:

| method        |    ΔBE |    exp |  error |
|---------------|-------:|-------:|-------:|
| PP-PBE        | 10.568 | 11.050 | −0.482 |
| PP-PBE(B3LYP) | 11.192 | 11.050 | +0.142 |

B3LYP gives +0.14 eV on an 11 eV shift — 1.3% relative, better than our published C-shift
MAE of 0.20 eV, and ~70% below the PP-PBE error of −0.48 eV. Gas-phase XPS resolves
~0.1 eV, so this is at the edge of what the measurement itself distinguishes.

## Gotchas

Inputs come from our own production runs in
`$ARES_DSCF_DATA/{2,15}/{I,F}` (the `DSCF_DATA/C` folder of the JCTC 2022 runs; 2 = CH₄, 15 = CF₄) —
`setup.py` copies those `POSCAR`s verbatim, and `analyze.py` reads their `ares.log`s. Both stop
with a message when `ARES_DSCF_DATA` (or, for `setup.py`, `ARES_PP_DIR`) is not set. Four things
that cost time to work out:

- **The grid is per molecule, not global.** CH₄ uses `GRIDSPACING=0.12` / `ISOrmax=5.0`,
  CF₄ `0.07` / `5.5`. CF₄'s `ares.in` is unreadable (mode 600); recover its settings from
  the `ares.log` header.
- **Use the `PPs_copy` pseudopotentials.** Those in `../CH4/` are a different, larger set
  (higher `l_max`) and do not reproduce the paper.
- **Grids differ from the 2022 logs** (102³/182³ vs 84³/158³) because that run used a
  different binary on Cori. Same physics — we reproduce ΔBE to ~1 meV.
- **`ares.log` lags badly while running** — NVFORTRAN buffers in 4 KB blocks, so a healthy
  job can look stalled. The job scripts fix this with `srun --unbuffered stdbuf -oL`.

Each molecule has one carbon, so the final state keeps species `C` and just binds
`C.pbe-mt-cpi-1s.UPF` with `DCHARGE=-1.0`.

## Layout

```
setup.py  run8.qsub  run_b3lyp_cf4.qsub  analyze.py
{pbe,b3lyp}/{CH4,CF4}/{IS,FS_1s}/
archive_2026-08-14_salvaged.md    numbers from earlier, superseded runs
```
