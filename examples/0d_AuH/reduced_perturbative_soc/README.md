Scaled-down AuH molecule (same geometry and pseudopotentials as
`examples/0d_AuH/`, reduced grid resolution for a fast, apples-to-apples
comparison) exercising perturbative spin-orbit coupling combined with
collinear spin polarization -- the mode `parsec_python.cli` runs whenever
`Spin_Polarization=true` and any species has `SO_PSP=true` but
`SO_from_scratch`/`SCF_SO` is not set.

`parsec.in` is the PARSEC.py-flavored input (used by
`tests/unit/test_0d_AuH_fortran_reference.py`). `parsec_fortran.in` is the
same physical system in Fortran PARSEC's own input, used to generate
`fortran_reference.out`:

```
mpiexec -n 2 /home/a2d2/Programs/Research/PARSEC/src/parsec-ubuntu_gnu-mpifort-gfortran-16.2.1.linux_ubuntu.mpi
```

(git commit `f432777`, config `ubuntu_gnu`; PARSEC_ACCELERATED environment
not involved -- run from a directory containing only `parsec_fortran.in`
renamed to `parsec.in` plus the two POTRE files.)

Converged scalar-relativistic (pre-SOC) spin-polarized total energy:
Fortran -68.32490203 Ry vs PARSEC.py -68.32491144 Ry (9.4e-6 Ry). Converged
perturbative-SOC state 1 eigenvalue/<S_z>: Fortran -0.818415 Ry / 0.9779 vs
PARSEC.py -0.818696 Ry / 0.9779. See the parent conversation's summary for
the full state-by-state comparison; states inside tightly-spaced
near-degenerate multiplets (Au's d-manifold) show larger <S_z> spread
between the two codes, expected basis-sensitivity rather than a
discrepancy.
