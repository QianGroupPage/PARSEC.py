# Initial structures from a force field: three examples

Each folder holds one DFT relaxation (BFGS) twice: from a crude structure as
written, and with one more line,

```text
Initial_Structure: mlff
```

which first relaxes that structure with the default force field, UMA
`uma-s-1p2`, and starts the DFT run from the result. Set up the force field
once per machine ([the MLFF guide](../../src/parsec_python/MLFF/README.md#getting-started));
the inputs name no interpreter, so `PARSEC_MLFF_PYTHON` must point to it.

| Folder | System | Crude structure | DFT criterion |
|---|---|---|---|
| `benzene/` | C6H6, 12 atoms | sketched: C-C 1.54, C-H 1.15 angstrom (DFT: about 1.39, 1.09) | `Force_Min` 0.002 Ry/bohr, SCF 1e-5 Ry |
| `C35H36/` | nanodiamond, 71 atoms | the GFN2-xTB structure of `examples/0d_nanodiamonds` expanded by 3 % | `Force_Min` 0.001 Ry/bohr, SCF 1e-6 Ry |
| `C185H124/` | nanodiamond, 309 atoms | the same, expanded by 3 % | `Force_Min` 0.001 Ry/bohr, SCF 1e-6 Ry |

The expansion keeps the point group of the cluster, so both runs use its
symmetry sectors; the force field keeps it too. The nanodiamond inputs are
those of the Fortran comparison of `examples/0d_nanodiamonds` (CA-LDA,
0.2 angstrom grid, PARSEC's Hartree boundary).

## Running them

From the repository root, on a workstation or laptop:

```bash
python src/parsec_python/main.py examples/0d_mlff_initial_structure/benzene/from_crude/parsec.in --no-archive
python src/parsec_python/main.py examples/0d_mlff_initial_structure/benzene/with_mlff/parsec.in --no-archive
```

and the same for `C35H36/from_expanded` and `C35H36/with_mlff`. On a GPU
node, for the larger cluster:

```bash
scripts/run_multi_gpu_scf.sh examples/0d_mlff_initial_structure/C185H124/from_expanded/parsec.in runs/C185H124_from_expanded
scripts/run_multi_gpu_scf.sh examples/0d_mlff_initial_structure/C185H124/with_mlff/parsec.in runs/C185H124_with_mlff
```

A run with the force field writes `mlff_structure.xyz`, `mlff_relaxation.xyz`
and `mlff.log` beside its input (the launch script: into its output
directory), and its `parsec.out` begins with what the force field did:
the model, its steps, the largest force before and after, the largest
displacement and the time. `atom.cor` lists the DFT structures. A second run
of the same input takes `mlff_structure.xyz` instead of relaxing again.

## What they gave

| System | Machine | DFT structures: crude / after `uma-s-1p2` | Time (program time of `parsec.out`): crude / force field + DFT |
|---|---|---:|---:|
| C6H6 | Windows laptop, RTX 5070 | 13 / 11 | 261 s / 132 + 132 s |
| C35H36 | Windows laptop, RTX 5070 | 29 / 13 | 658 s / 133 + 302 s |
| C35H36 | one node, 4 A100 | 29 / 13 | 33 s / 30 + 22 s |
| C185H124 | one node, 4 A100 | 38 / 18 | 122 s / 19 + 57 s |

Every pair ends at the same structure and energy. From the GFN2-xTB
structures themselves the DFT runs took 13 (C35H36) and 21 (C185H124)
structures: the force field brings a crude structure back to that quality.
The force field costs a fixed half minute to two minutes, mostly loading the
model (about two minutes on the Windows laptop, where the files are scanned
on first use), and it pays off once the DFT run is expensive: on four A100
from a few hundred atoms on, on the laptop GPU, whose double precision is
slow, already for C35H36 (a third less time). The larger clusters of
[the MLFF guide](../../src/parsec_python/MLFF/README.md#what-it-gains-and-which-model)
saved 37 to 49 %. The program time at the end of `parsec.out` leaves the
force field out; its time is in the lines that report it.
