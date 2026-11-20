# H2 with PARSEC's canonical pseudopotential (no double grid)

This directory preserves the full canonical hydrogen pseudopotential copied
from:

```text
<PARSEC_FORTRAN_ROOT>/tests/H2/H_POTRE.DAT
```

That file contains 863 radial points. It is unrelated to the six-point
synthetic fixture under `tests/data`, which exists only for fast unit tests.

The stock PARSEC H2 input uses `Double_Grid_Order: 3`. The Python solver does
not yet implement Ono-Hirose double-grid integration, so this input explicitly
sets the order to one.

## Run the Python calculation

From the repository root:

```powershell
python src\parsec_python\main.py examples\h2_canonical_nodg\parsec.in --no-archive
```

The calculation writes a PARSEC-style text report to `parsec.out` in
this directory. The retained `parsec_python.out` is this command with
`--log parsec_python.out`, run with the code of 19 November 2026 on a laptop
GPU (RTX 5070; paths shortened to repository paths). H2 on the z axis has the
point group D4h on the grid (16 signed permutations); since 18 November 2026
(`PARSEC_ORBITAL_GROUP_MAX_COST=0.75`) the orbitals use the six weighted
one-dimensional sectors of a subgroup of order 8 (weights 1, 1, 2, 1, 1, 2),
where the run of 5 November used the eight sectors of the abelian D2h. With
the CHEBFF-first/SUBSPACE-later eigensolver the calculation converged in 17
SCF iterations (20 on the D2h route of 5 November, with the same total
energy):

| Quantity | Python result |
|---|---:|
| Total energy | -2.13713415 Ry |
| Occupied eigenvalue | -0.7196461994 Ry |
| Plain SRE | 0.0031773149 |
| Charge-weighted SRE | 0.0000777048 |

The eigenvalue table lists 17 levels for `States_Num: 16`: a weighted sector
reports every partner of its last multiplet, here the second partner of the
degenerate pair 16-17. `PARSEC_ORBITAL_GROUP_MAX_COST=0.5` gives the D2h
route of before (16 levels).

The report follows PARSEC's organization and labels for the grid, atoms,
pseudopotential, eigenstates, SCF iterations, energy decomposition, and SRE
values. It identifies itself as `PARSEC-PYTHON` and explicitly lists the
PARSEC-only results that this implementation does not calculate.

## Comparison with the stock PARSEC H2 result

The following comparison is useful as a numerical check, but it is **not a
strict reproduction test** because the double-grid orders differ:

| Quantity | Python, DG=1 | Stock PARSEC, DG=3 | Python - PARSEC |
|---|---:|---:|---:|
| Total energy [Ry] | -2.13713415 | -2.13713410 | -0.00000005 |
| Occupied eigenvalue [Ry] | -0.7196461994 | -0.7194900830 | -0.0001561164 |
| Eigenvalue energy [Ry] | -1.43929240 | -1.43898017 | -0.00031223 |
| Hartree energy [Ry] | 2.49259088 | 2.49240374 | 0.00018714 |
| Integral of `Vxc*rho` [Ry] | -1.63991186 | -1.63979721 | -0.00011465 |
| Exchange-correlation energy [Ry] | -1.25644225 | -1.25635504 | -0.00008721 |
| Electron-ion energy [Ry] | -6.74648299 | -6.74608367 | -0.00039932 |
| Ion-ion energy [Ry] | 1.41113867 | 1.41113867 | 0.00000000 |

The total energy is stationary; the separate terms and the occupied level
follow the potential at which each SCF stops within the criterion of
2e-4 Ry (the Python level moves by 8e-5 Ry between its last two iterations).

PARSEC currently segfaults in its `nonloc` routine before SCF when this
one-channel H potential is run with double-grid order one, both on the WSL
native filesystem and on the Windows-mounted filesystem. Therefore this
directory does not contain a valid `parsec_reference.out` and must not be used
for an order-one reproduction claim. The near agreement with stock PARSEC is
encouraging, but a controlled canonical comparison requires implementing the
Ono-Hirose order-three double grid and then using PARSEC's unmodified input.
