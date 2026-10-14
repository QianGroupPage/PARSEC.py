Scaled-down AuH molecule (same geometry and pseudopotentials as
`examples/0d_AuH/`, reduced grid resolution) exercising self-consistent
spin-orbit coupling.

`parsec.in` (PARSEC.py-flavored: `SO_from_scratch=true`, no
`Spin_Polarization`, since PARSEC.py's cluster self-consistent SOC
currently only supports running without collinear spin polarization) is
used by `tests/unit/test_0d_AuH_fortran_reference.py`. `parsec_fortran.in`
is the same physical system in Fortran PARSEC's own input, used to
generate `fortran_reference.out`:

```
mpiexec -n 2 /home/a2d2/Programs/Research/PARSEC/src/parsec-ubuntu_gnu-mpifort-gfortran-16.2.1.linux_ubuntu.mpi
```

(git commit `f432777`, config `ubuntu_gnu`; run from a directory containing
only `parsec_fortran.in` renamed to `parsec.in` plus the two POTRE files.)

Note: Fortran PARSEC's own input parser hard-rejects (`ierr=151`,
`usrinputfile.F90:2574-2580`) any species with `SO_PSP=true` unless
`Spin_Polarization=true` is also set -- this holds even for `SCF_SO`, so
`parsec_fortran.in` sets `Spin_Polarization: .true.` despite PARSEC.py's
side not using it. AuH's ground state has zero net magnetic moment in both
codes, so the comparison is still meaningful: the converged self-consistent
SOC total energy matches to 8.4e-6 Ry (Fortran -68.34107156 Ry vs
PARSEC.py -68.34106315 Ry), and Kramers double-degeneracy holds to high
precision in both. Individual eigenvalues agree to ~1e-3 Ry; <S_z> within
near-degenerate pairs diverges more, same basis-sensitivity caveat as the
perturbative case.
