# Benzene PARSEC/Python benchmark

This directory contains the unchanged benzene example input and
pseudopotentials copied from:

```text
/home/zeyizhang/PARSEC/examples/benchmarks/0d_benzene
```

The copied files are byte-for-byte identical to the WSL sources:

| File | SHA-256 |
|---|---|
| `parsec.in` | `c213ed4eceef790e7bdb86d6f63fc49e513af82a785a01a7b7011f54adb44764` |
| `C_POTRE.DAT` | `2e073cb81f66327a15d82a3955b8822c72f13494d31ec51799b2b3c122598c8c` |
| `H_POTRE.DAT` | `202df2aba56d7aa99a515909add249b643cde27f1aff67bc79b4ceeebf2af191` |

The system has six carbon and six hydrogen atoms, 30 valence electrons,
20 requested states, a 16-bohr spherical domain, 0.4-bohr spacing, and the
default order-12 finite-difference stencil. The omitted double-grid order
defaults to one in PARSEC, so no input change was required.

Carbon uses its `p` channel as the local potential. Its `s` channel supplies
one Kleinman-Bylander projector per carbon atom, giving six nonlocal projector
columns. Hydrogen has only its local `s` channel. Neither potential has a
nonlinear core correction.

## Run

From `src/new_architecture`:

```powershell
python main.py benchmarks\0d_benzene\parsec.in --dry-run
python main.py benchmarks\0d_benzene\parsec.in --no-archive
```

The physical calculation writes `parsec.out` in this directory. The
completed native CHEBFF/SUBSPACE run used the full 268,096-point grid and
converged in 15 SCF iterations:

| Quantity | Python |
|---|---:|
| Total energy | -75.38863311 Ry |
| Charge-weighted SRE | 0.0000947232 |
| SCF time | 410.91 s |
| Total wall time | 411.84 s |

## PARSEC reference

The upstream example did not include a reference output. A new reference was
therefore calculated from the unchanged WSL files with the matching PARSEC
OpenMPI build and eight MPI ranks. Copies are stored here as
`parsec_reference.out` and `parsec_reference_time.txt`.

PARSEC found the eight-operation `D_2h` subgroup and reduced the Hamiltonian
from 268,096 to 33,512 grid points per representation. It converged in 17 SCF
iterations to `-75.38863314 Ry`. The Python-minus-PARSEC total-energy
difference is `+0.00000003 Ry`.

See `COMPARISON.md` for the energy components, frontier eigenvalues, convergence
values, and timing caveats.
