# H2 with PARSEC's canonical pseudopotential (no double grid)

This directory preserves the full canonical hydrogen pseudopotential copied
from:

```text
/home/zeyizhang/PARSEC/tests/H2/H_POTRE.DAT
```

That file contains 863 radial points. It is unrelated to the six-point
synthetic fixture under `tests/data`, which exists only for fast unit tests.

The stock PARSEC H2 input uses `Double_Grid_Order: 3`. The Python solver does
not yet implement Ono-Hirose double-grid integration, so this input explicitly
sets the order to one.

## Run the Python calculation

From `src/new_architecture`:

```powershell
python main.py benchmarks\h2_canonical_nodg\parsec.in --no-archive
```

The calculation writes a PARSEC-style text report to `parsec.out` in
this directory. With the native CHEBFF-first/SUBSPACE-later implementation,
the completed calculation converged in 21 SCF iterations:

| Quantity | Python result |
|---|---:|
| Total energy | -2.13713415 Ry |
| Occupied eigenvalue | -0.7196931667 Ry |
| Plain SRE | 0.0131434491 |
| Charge-weighted SRE | 0.0000846232 |

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
| Occupied eigenvalue [Ry] | -0.7196931667 | -0.7194900830 | -0.0002030837 |
| Eigenvalue energy [Ry] | -1.43938633 | -1.43898017 | -0.00040616 |
| Hartree energy [Ry] | 2.49249032 | 2.49240374 | 0.00008658 |
| Integral of `Vxc*rho` [Ry] | -1.63985025 | -1.63979721 | -0.00005304 |
| Exchange-correlation energy [Ry] | -1.25639533 | -1.25635504 | -0.00004029 |
| Electron-ion energy [Ry] | -6.74628231 | -6.74608367 | -0.00019864 |
| Ion-ion energy [Ry] | 1.41113867 | 1.41113867 | 0.00000000 |

PARSEC currently segfaults in its `nonloc` routine before SCF when this
one-channel H potential is run with double-grid order one, both on the WSL
native filesystem and on the Windows-mounted filesystem. Therefore this
directory does not contain a valid `parsec_reference.out` and must not be used
for an order-one reproduction claim. The near agreement with stock PARSEC is
encouraging, but a controlled canonical comparison requires implementing the
Ono-Hirose order-three double grid and then using PARSEC's unmodified input.
