# Meta-GGAs of bulk Si

`Si_FHIPP.DAT` is the FHI98PP potential of the library (`src/pp_generation/NCPPs/PBE/Si/FHI/Si_FHIPP.DAT`,
Troullier-Martins, PBE, no core correction), the one of the Quantum ESPRESSO 7.5 + libxc meta-GGA references compared
below. The check script writes the fcc cell of those references (a = 10.26 bohr, two atoms,
`States_Num 8`, `Fermi_Temp 10 K`, Gamma-centred `Monkhorst_Pack_Grid n n n` reduced to its irreducible wedge) with a
functional, grid spacing and mesh, runs the k-point loop (on the GPUs where CuPy finds them) and prints energy, band
edges, gap, iterations and timings as JSON:

```bash
python -m parsec_python.acceleration.benchmarks.meta_gga_periodic_check --functional pbe,scan,r2scan,tpss \
    --kmesh 1,2,4 --spacing 0.20 [--gradient density|orbital] [--term-timing] [--program] --out results
```

`r2scan` and `tpss` are libxc's (`LibXC_Functional`, set `PARSEC_LIBXC_LIBRARY`); `scan` is the native SCAN. The
results against Quantum ESPRESSO are in `src/parsec_python/README.md` ("Meta-GGA functionals", "Periodic cells").
