# Water molecule: BFGS geometry optimization

The geometry-optimization example `0d_water_geoopt` of Fortran PARSEC,
unchanged: a water molecule in a sphere of 12 bohr on a 0.3 bohr grid, CA-LDA,
`Minimization: BFGS`, `Force_Min: 0.02` Ry/bohr.

```powershell
python src\parsec_python\main.py examples\0d_water_geoopt\parsec.in --no-archive
```

The run computes seven structures in about 20 s on a laptop GPU and ends with
exit status 0. Beside the input it writes `parsec.out`, `atom.cor`,
`relax_restart.dat`, `iterate.dat` and `relaxation.xyz`; see
[Forces, relaxation and molecular dynamics](../../src/parsec_python/FORCES_AND_RELAXATION.md).

## Expected trajectory

`atom_cor_fortran_reference.out` is the `atom.cor` of Fortran PARSEC for this
input. Energy per atom in eV and largest force in Ry/bohr of every structure:

| Structure | Fortran PARSEC | This package |
|---:|---|---|
| 0 | -155.04550, 0.22661 | -155.04550, 0.22661 |
| 1 | -154.56606, 0.21984 | -154.56607, 0.21984 |
| 2 | -155.26126, 0.08884 | -155.26126, 0.08884 |
| 3 | -155.31577, 0.08506 | -155.31577, 0.08506 |
| 4 | -155.42371, 0.10606 | -155.42368, 0.10614 |
| 5 | -155.46132, 0.03881 | -155.46129, 0.03882 |
| 6 | -155.47162, 0.01352 | -155.47162, 0.01352 |

The input leaves the SCF criterion at PARSEC's default of 2e-4 Ry, so the two
codes stop their SCF loops at slightly different points and the last digits
differ. With `Convergence_Criterion: 1.0e-8` and `Max_Iter: 200` added, all
seven energies agree in every printed digit, the largest forces within 1e-5
Ry/bohr and the coordinates within 1e-6 bohr.

The second structure lies higher in energy than the first: the first step of
L-BFGS-B moves along the force by the inverse of its norm, and the line search
then shortens it. Fortran PARSEC does the same.

## Stopping and continuing

```text
Movement_Num: 2
```

stops the run after two movements with exit status 4 (not relaxed). A second
run with

```text
Relax_Restart: true
```

or with `--restart-from relax_restart.dat` on the command line continues from
the saved structure and optimizer state. The file is PARSEC's: either code
continues a run of the other.
