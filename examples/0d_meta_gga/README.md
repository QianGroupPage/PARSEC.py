# Pseudopotentials of the meta-GGA checks

Copies of files of the project's library `src/pp_generation/NCPPs/PBE` (PBE
norm-conserving pseudopotentials), kept here because that library does not
travel with every run:

| file | from | used by |
|---|---|---|
| `H_FHIPP.DAT`, `C_FHIPP.DAT`, `N_FHIPP.DAT`, `O_FHIPP.DAT` | `NCPPs/PBE/<element>/FHI` | `python -m parsec_python.acceleration.benchmarks.meta_gga_check` (H2O, CH4, N2, CO, NH3, O2; the files of the Quantum ESPRESSO comparison), `tests/unit/test_meta_gga_scf.py` |
| `H_POTRE.DAT`, `Li_POTRE.DAT` | `NCPPs/PBE/<element>/PARSEC` | `tests/unit/test_meta_gga_scf.py` (LiH: Li has a nonlinear core correction) |

A meta-GGA reads PBE pseudopotentials (see "Meta-GGA functionals" in
`src/parsec_python/README.md`).
