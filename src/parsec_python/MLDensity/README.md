# Machine-learned initial densities: tutorial and reference

PARSEC.py can start the SCF of an isolated (cluster) calculation from a valence
density predicted by a neural network instead of the superposition of atomic
densities (SAD). Two published models are built in, **ChargE3Net** and **SCDP**,
and any other model can be connected without changing PARSEC.py. Only the
starting density changes: the grid, pseudopotentials, Hamiltonian, mixing,
convergence test and final energy are those of an ordinary calculation, so a
converged result does not depend on the model.

```text
parsec.in ──► atoms + DFT grid ──► request.npz ──► bridge script, run by the   ──► density.npz ──► SCF
                                                   model's own Python (GPU)
```

Each model keeps its own Python environment (PyTorch, e3nn, ...); PARSEC.py
does not import any of it. It writes the atoms and the grid points to a small
file, runs a *bridge* script of the model in the model's environment, and
reads the density back.

**Contents**

1. [Which model, and when it helps](#1-which-model-and-when-it-helps)
2. [Install the models (once)](#2-install-the-models-once)
3. [Check the installation](#3-check-the-installation)
4. [Use it](#4-use-it): [A. in `parsec.in`](#a-one-line-in-parsecin) ·
   [B. predict first](#b-predict-first-then-run) ·
   [C. any XYZ on a box you choose](#c-any-xyz-file-on-a-box-you-choose) ·
   [D. from Python](#d-from-python)
5. [Connect another density model](#5-connect-another-density-model)
6. [Reference](#6-reference)
7. [Troubleshooting](#7-troubleshooting)

## 1. Which model, and when it helps

| `Initial_Density` | `ML_Density_Model` | Network | Trained on | Weights |
|---|---|---|---|---|
| `charge3net` | `qm9` (default) | ChargE3Net, E(3)-equivariant, evaluated at every point | QM9: small molecules of H, C, N, O, F | shipped with the repository |
| `charge3net` | `mp`, `nmc` | the same, periodic training sets | Materials Project, NMC cathodes | shipped; PARSEC.py runs them only for isolated systems |
| `scdp` | `fast` (default) | SCDP eSCN, 4 layers, no virtual nodes | QM9 | [Zenodo 13146215](https://zenodo.org/records/13146215) |
| `scdp` | `accurate` | SCDP eSCN, 8 layers, bond virtual nodes | QM9 | the same archive |

SCF iterations from each start, measured on an A100 for four molecules of the
[regression suite](../../../examples/ml_initial_density/README.md) (PBE,
0.2 Å grid, 7 Å spheres for the small and 9 Å for the aromatic molecules,
CHEBDAV, `2e-4 Ry`; the models predicted at the exact grid points):

| Molecule | SAD | ChargE3Net `qm9` | SCDP `fast` | SCDP `accurate` |
|---|---:|---:|---:|---:|
| H2O | 13 | 9 | 12 | 11 |
| CH4 | 18 | 7 | 10 | 7 |
| C6H6 | 14 | 15 | 9 | 9 |
| C10H8 | 13 | 8 | 7 | 8 |

All final energies agree with the SAD start within 1.1e-6 Ry. Electrons in
the raw predictions, before normalization (8, 8, 30, 48 expected):
ChargE3Net 8.04, 7.98, 29.93, 47.91; SCDP fast 7.21, 8.03, 29.65, 47.89;
SCDP accurate 8.54, 8.00, 30.00, 48.00. The suite's water is stretched
(O–H 1.10 Å instead of 0.96 Å), and both SCDP models miss its electron count;
for water at equilibrium SCDP fast gives 7.77 and accurate 7.96 electrons.

What to expect:

- **Training domain.** The molecular checkpoints were trained on QM9:
  molecules of up to nine heavy atoms (C, N, O, F) with hydrogen, near
  equilibrium. In that domain a predicted start usually saves SCF
  iterations, though not always (the table). SCDP refuses other elements.
- **SCDP and molecule size.** SCDP passes messages between all atoms within
  6 Å, and beyond QM9 size it leaves its training domain entirely: for the
  71-atom nanodiamond C35H36 its fast model predicts 58,000 electrons instead
  of 176, for C87H76 two million instead of 424. Use SCDP for QM9-sized
  molecules only. ChargE3Net evaluates each point from the atoms within 4 Å;
  validate larger systems against SAD as well. Whatever the model,
  PARSEC.py warns when a predicted density holds more than 10 % too many or
  too few electrons, since normalization then hides a poor prediction.
- **Time.** The network itself is fast: on an RTX 5070 laptop GPU ChargE3Net
  needs 3 s for water on a 180,000-point grid, on an A100 7–9 s for
  naphthalene on 382,000 points, and SCDP a few seconds. Starting the model
  costs more: 12 s on the laptop, 40–170 s on Perlmutter, where Python
  imports its packages from the project file system (section 2). For one
  small calculation the saved iterations can be worth less than this; a
  prediction is cached, and predicting first (section 4B) pays the start
  once for any number of runs.
- Isolated systems only, and not spin-polarized: a spin-polarized calculation
  starts from SAD with `Initial_Spin_Polarization` (an ML density has no spin
  densities). Periodic inputs refuse an ML start.
- A core-hole species such as `C-1s` is given to the model as `C`; the model
  does not know the core hole, so this is an approximate start.

## 2. Install the models (once)

Each model gets its own environment, because their dependencies conflict
(SCDP needs e3nn 0.4.4, ChargE3Net a newer e3nn). Put them in a folder
`parsec_ml_workspace` **next to** the PARSEC.py checkout; then PARSEC.py finds
everything without any path in `parsec.in`:

```text
work/
├── parsec_python/              this repository (DFT environment .venv312 or yours)
└── parsec_ml_workspace/
    ├── charge3net/             git clone + .venv/
    └── scdp/                   git clone + .venv/ + models/share_models/...
```

The recipes below were run as written on Linux (a Perlmutter login node,
inference on A100 nodes) for both models, and on Windows 11 (RTX 5070 laptop
GPU) for ChargE3Net. They use PyTorch 2.11 with CUDA 12.8 for both models,
including SCDP, whose own `requirements.txt` pins torch 1.13.1+cu116: that
old build has no kernels for recent GPUs, and the bridges handle the loading
changes of newer PyTorch. Choose another index for another CUDA version, or
`https://download.pytorch.org/whl/cpu` for a CPU-only machine. SCDP needs
Python 3.10 or 3.11: its pinned `basis_set_exchange==0.9.1` is a source
package that no longer builds on Python 3.12.

Versions that ran: ChargE3Net with Python 3.12, torch 2.11.0, e3nn 0.6.0
(e3nn 0.5.1 gave the same densities but took three times as long); SCDP
with Python 3.10, torch 2.11.0, e3nn 0.4.4, lightning 2.1.3.

### Linux and HPC clusters

[uv](https://docs.astral.sh/uv/) downloads the right Python versions and
installs the large packages quickly (`pip` works as well):

```bash
cd work                                   # the folder that holds parsec_python
mkdir -p parsec_ml_workspace && cd parsec_ml_workspace
curl -LsSf https://astral.sh/uv/install.sh | sh     # or: pip install --user uv
TORCH=https://download.pytorch.org/whl/cu128

# ChargE3Net (its checkpoints are part of the repository)
git clone https://github.com/AIforGreatGood/charge3net.git
uv venv --python 3.12 charge3net/.venv
uv pip install --python charge3net/.venv/bin/python torch==2.11.0 --index-url $TORCH
grep -v -E '^(torch([=<>~ ]|$)|--extra-index-url)' charge3net/requirements.txt > charge3net-requirements.txt
uv pip install --python charge3net/.venv/bin/python -r charge3net-requirements.txt

# SCDP (weights from Zenodo, 1.9 GB for both models)
git clone https://github.com/kyonofx/scdp.git
uv venv --python 3.10 scdp/.venv
uv pip install --python scdp/.venv/bin/python torch==2.11.0 --index-url $TORCH
grep -v -E '^(torch([=<>~ ]|$)|--extra-index-url)' scdp/requirements.txt > scdp-requirements.txt
uv pip install --python scdp/.venv/bin/python -r scdp-requirements.txt
mkdir -p scdp/models
curl -L -o scdp/models/scdp_pretrained_models.tar.gz \
  https://zenodo.org/records/13146215/files/scdp_pretrained_models.tar.gz
tar -xzf scdp/models/scdp_pretrained_models.tar.gz -C scdp/models
```

On a cluster, install on a login node into a project or software file
system (each environment takes about 7 GB, the SCDP weights 2 GB), and set
`UV_CACHE_DIR` there if the home quota is small. Inference then runs on a GPU
node. Python imports thousands of small files: from the Perlmutter project
file system (CFS) loading SCDP took about half a minute before any
prediction. NERSC recommends `/global/common/software` for software
environments.

### Windows

Use the python.org interpreters through the `py` launcher (with Smart App
Control, interpreters that are not signed, such as those uv downloads, can
be blocked; see [Troubleshooting](#7-troubleshooting)). SCDP needs Python
3.10 or 3.11 (`py -0` lists what is installed; python.org has installers up
to 3.10.11 and 3.11.9). The ChargE3Net part was run as written; the SCDP part
follows the same steps but was not run on Windows. In PowerShell:

```powershell
cd C:\work
mkdir parsec_ml_workspace; cd parsec_ml_workspace
$torch = "https://download.pytorch.org/whl/cu128"

git clone https://github.com/AIforGreatGood/charge3net.git
py -3.12 -m venv charge3net\.venv
charge3net\.venv\Scripts\python -m pip install torch==2.11.0 --index-url $torch
Get-Content charge3net\requirements.txt |
  Where-Object { $_ -notmatch '^(torch([=<>~ ]|$)|--extra-index-url)' } |
  Set-Content charge3net-requirements.txt
charge3net\.venv\Scripts\python -m pip install -r charge3net-requirements.txt

git clone https://github.com/kyonofx/scdp.git
py -3.10 -m venv scdp\.venv
scdp\.venv\Scripts\python -m pip install torch==2.11.0 --index-url $torch
Get-Content scdp\requirements.txt |
  Where-Object { $_ -notmatch '^(torch([=<>~ ]|$)|--extra-index-url)' } |
  Set-Content scdp-requirements.txt
scdp\.venv\Scripts\python -m pip install -r scdp-requirements.txt
mkdir scdp\models
curl.exe -L -o scdp\models\scdp_pretrained_models.tar.gz `
  https://zenodo.org/records/13146215/files/scdp_pretrained_models.tar.gz
tar -xzf scdp\models\scdp_pretrained_models.tar.gz -C scdp\models
```

### Somewhere else

Any location works. Name it in `parsec.in` (section 4A), on the command line
(`--repository`, `--python`, `--checkpoint`), or through the environment
variables `CHARGE3NET_REPO`, `CHARGE3NET_PYTHON`, `SCDP_REPO`, `SCDP_PYTHON`,
or `PARSEC_ML_WORKSPACE` (a folder with `charge3net/` and `scdp/`). An
environment that is not a `.venv` inside the clone (conda, a module) is
named by its interpreter, e.g. `ML_Density_Python: /path/to/env/bin/python`.

## 3. Check the installation

All commands run with the PARSEC.py environment from the PARSEC.py checkout;
the models run in their own environments.

```bash
python src/parsec_python/ml_density.py models    # what is found where (runs nothing)
python src/parsec_python/ml_density.py check     # runs each model once on a water molecule
```

`check` prints the bridge, repository, interpreter and checkpoint it uses and
where each came from, then predicts the density of water on a small box:

```text
== charge3net
  bridge     C:\work\parsec_python\src\parsec_python\MLDensity\bridges\charge3net.py  [built in]
  repository C:\work\parsec_ml_workspace\charge3net  [PARSEC_ML_WORKSPACE]
  python     C:\work\parsec_ml_workspace\charge3net\.venv\Scripts\python.exe  [repository .venv]
  checkpoint C:\work\parsec_ml_workspace\charge3net\models\charge3net_qm9.pt  [found in the repository]
  ChargE3Net qm9 on NVIDIA GeForce RTX 5070 Laptop GPU (torch 2.11.0+cu128), loaded in 40.4 s; 20,088 points, 17,868 within 4 A of an atom; 21,468 edges per call
  done in 41.9 s
  ChargE3Net qm9: 20,088 points in 43.2 s on NVIDIA GeForce RTX 5070 Laptop GPU
  electrons in the prediction: 7.9925
  840 negative values (min -5.480e-07 e/bohr^3); the SCF import clips them and normalizes the integral
  OK
== scdp
  ...
```

(The first run of a fresh environment loads slowly; later runs load in about
12 s on this machine.)

`OK` means that the model ran on the device shown and its density holds
water's 8 valence electrons (within 10 %). Check one model with
`--provider scdp`, and add `--repository/--python/--checkpoint/--model` for
an installation outside the workspace.

## 4. Use it

The commands below use the water example in
[`examples/ml_initial_density/quickstart`](../../../examples/ml_initial_density/quickstart/parsec.in).
Copy it to a work folder first, so that its results stay out of the checkout:

```bash
cp -r examples/ml_initial_density/quickstart ~/water     # PowerShell: Copy-Item -Recurse
PP=$PWD/examples/ml_initial_density/pseudopotentials
```

### A. One line in `parsec.in`

Replace `Initial_Density: sad` by the model:

```text
Initial_Density: charge3net          # or: scdp
```

and run the calculation as always:

```bash
python src/parsec_python/main.py ~/water/parsec.in --pp-dir $PP
```

Before the SCF, PARSEC.py runs the model at the exact points of the
calculation's grid, writes the prediction to `.parsec_ml_density_cache/` next
to `parsec.in`, checks it, clips small negative values, normalizes it to the
valence-electron count of the pseudopotentials, and starts the SCF from it.
`parsec.out` reports `Initial density method = charge3net` and the time of
the prediction (`Initial valence-density setup`). Running the same geometry
and grid again reuses the cached prediction.

If the models are not in `parsec_ml_workspace`, add their paths:

```text
Initial_Density: scdp
ML_Density_Model: accurate                       # default: fast
ML_Density_Repository: /path/to/scdp
ML_Density_Python: /path/to/scdp/.venv/bin/python
ML_Density_Checkpoint: /path/to/scdp/models/share_models/qm9_bond_K8L6_beta_1.3_vo/epoch=37-step=294120.ckpt
```

ChargE3Net finds `models/charge3net_<model>.pt` in its repository by itself;
SCDP finds a checkpoint below its `models/` folder, telling the fast from the
accurate model by the archive's directory names (`qm9_none_...`,
`qm9_bond_...`). Relative paths are relative to `parsec.in`.

### B. Predict first, then run

`predict` runs only the model, shows its progress, and writes the density
file a calculation imports:

```bash
python src/parsec_python/ml_density.py predict --input ~/water/parsec.in --pp-dir $PP
```

```text
ChargE3Net qm9: exact grid of parsec.in, 179,944 points
  ChargE3Net qm9 on NVIDIA GeForce RTX 5070 Laptop GPU (torch 2.11.0+cu128), loaded in 11.5 s; 179,944 points, 46,740 within 4 A of an atom; 21,468 edges per call
  20% of the points (12 s)
  50% of the points (13 s)
  80% of the points (14 s)
  done in 14.2 s
ChargE3Net qm9: 179,944 points in 15.3 s on NVIDIA GeForce RTX 5070 Laptop GPU
  electrons in the prediction: 8.0448 (valence electrons of the calculation: 8)
  20 negative values (min -1.628e-07 e/bohr^3); the SCF import clips them and normalizes the integral
  wrote /home/me/water/charge3net_qm9_density.npz
To start an SCF from it, put in parsec.in:
  Initial_Density: charge3net
  ML_Density_File: charge3net_qm9_density.npz
```

Add the two printed lines to `parsec.in` (or keep `Initial_Density` and add
`ML_Density_File`). The SCF then needs neither the model nor its
environment: predict on a GPU machine, copy the `.npz` with the input, and
run anywhere. This is the better way for large systems (the prediction
shows its progress, and its cost is separate from the SCF), for job scripts
on clusters, and for reusing one prediction in many calculations.

The file stores the density at every grid point with the point coordinates,
the atoms, the model, and its checkpoint. On import PARSEC.py checks that the
points are those of the calculation, and that the atoms are in the same frame
(not shifted against the grid).

Options: `--provider` and `--model` choose another model than the input's,
`-o file.npz` names the output, and `--cube file.cube` also writes a
[Gaussian cube](#file-formats) file to look at in VESTA, VMD, Avogadro, or
ASE. `--device cpu` runs the model on the CPU.

### C. Any XYZ file on a box you choose

The density can also be predicted for any molecule on a regular box,
independent of a PARSEC input: to look at it, to use it in another code, or
to give PARSEC.py a density made elsewhere. Lengths are in ångström. Give two
of box size, points per direction, and spacing:

```bash
cat > water.xyz <<'EOF'
3
water
O   0.0000   0.0000   0.1173
H   0.0000   0.7572  -0.4692
H   0.0000  -0.7572  -0.4692
EOF

# a box of 14 x 14 x 14 Å with 70 points per direction (spacing 0.2 Å)
python src/parsec_python/ml_density.py predict --xyz water.xyz --provider charge3net \
    --box 14 --points 70 --cube water.cube

# 5 Å of vacuum around the atoms on every side, spacing 0.15 Å
python src/parsec_python/ml_density.py predict --xyz water.xyz --provider scdp \
    --vacuum 5 --spacing 0.15

# three values set each direction separately
python src/parsec_python/ml_density.py predict --xyz water.xyz --provider scdp \
    --box 12 14 16 --points 60 70 80
```

| Given | Then |
|---|---|
| `--box L` and `--points N` | spacing `L/N` |
| `--box L` (or `--vacuum V`) and `--spacing h` | `N = ceil(L/h)` points; the box grows to `N h` |
| `--points N` and `--spacing h` | box `N h` |
| only `--vacuum`, or nothing | `--vacuum 5`, `--spacing 0.2` |

The points are placed symmetrically about the midpoint of the atoms'
bounding box, which is where a PARSEC.py calculation puts the origin (it
recentres the atoms unless `Minimization: manual`); `--no-center` keeps the
XYZ coordinates instead. The output `water_charge3net_qm9_box.npz` holds the
density on the box with its origin and spacing; a calculation of the same
atoms imports it with linear interpolation:

```text
Initial_Density: charge3net
ML_Density_File: water_charge3net_qm9_box.npz
ML_Density_Interpolation: linear
```

For a PARSEC.py calculation, B is better than C: a box whose points do not
lie on the calculation's grid is interpolated, which flattens the sharp
density peaks near the nuclei. In the
[regression suite](../../../examples/ml_initial_density/README.md) the stored
water density came from a box 0.087 Å off the grid: its peak is 10 % lower
than at the grid points (1.007 against 1.123 e/bohr³), and the SCF needed 13
iterations from it against 9 from the prediction at the grid points. A box
whose spacing is `Grid_Spacing` and whose points fall on the grid is imported
exactly.

### D. From Python

```python
import numpy as np
from parsec_python.MLDensity import box_around_atoms, predict_density

bohr = 1 / 0.529177210903
atoms = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]) * bohr
box = box_around_atoms(atoms, vacuum_bohr=4 * bohr, spacing_bohr=0.2 * bohr)
result = predict_density("scdp", ["O", "H", "H"], atoms, box.points(),
                         lower_bohr=box.lower_bohr, cell_bohr=box.lengths_bohr)
density = result.density.reshape(box.shape)      # electrons / bohr^3
print(box.integrate(result.density), result.seconds, result.metadata["device"])
```

`predict_density` takes any points (bohr) and the same model options as the
command line (`model=`, `repository=`, `python=`, `checkpoint=`, `bridge=`,
`device=`). `save_box_density`, `save_point_density`, and `write_cube` write
the files of sections B and C.

## 5. Connect another density model

### Without changing PARSEC.py: `Initial_Density: custom`

1. Copy [`bridges/template.py`](bridges/template.py) and replace its function
   `predict_density` with a call to your model. The template is a complete,
   working bridge (a toy model of atomic 1s densities); its docstring is the
   specification.
2. Point a calculation at it:

   ```text
   Initial_Density: custom
   ML_Density_Bridge: my_model_bridge.py
   ML_Density_Python: /path/to/my-model-env/bin/python
   ML_Density_Model: my-model-v1             # a label, passed on as --model
   ML_Density_Checkpoint: /path/to/weights   # optional, passed on as --checkpoint
   ML_Density_Repository: /path/to/code      # optional, the working directory
   ```

3. Test it the same way as the built-in models:

   ```bash
   python src/parsec_python/ml_density.py check --provider custom \
       --bridge my_model_bridge.py --python /path/to/my-model-env/bin/python
   ```

Everything else (`predict`, boxes, cube files, the cache, the frame checks,
the GPU driver) works for a custom bridge as for the built-in ones.

The protocol in short. The bridge is called as

```text
python bridge.py --request R.npz --output O.npz --model NAME --device auto|cpu|cuda
                 --chunk-size N [--repository DIR] [--checkpoint FILE]
```

| Request `R.npz` | |
|---|---|
| `symbols` (N,) | chemical symbols |
| `atom_positions_angstrom` (N, 3) | atoms |
| `probe_positions_angstrom` (M, 3) | the points where the density is wanted |
| `cell_angstrom` (3, 3) | an orthorhombic box holding atoms and points |
| `periodic` () | `False` |
| `target_coordinates_bohr` (M, 3) | the points in the frame of the calculation |

| Output `O.npz` | |
|---|---|
| `density` (M,) | volume density at the points, in their order |
| `coordinates_bohr` (M, 3) | `target_coordinates_bohr`, copied |
| `units` () | `e_per_angstrom3` or `e_per_bohr3` |
| `provider` (), `metadata_*` () | strings for the record (device, model, ...) |

Atoms and points are translated together into the positive box; a
translation-invariant molecular model needs nothing else. Exit with a
nonzero status on failure; lines starting with `progress:` are shown while
the model runs. Evaluate the model at the points given rather than on a box
of its own: interpolation degrades the start (section 4C).

### As a built-in model

Add `bridges/<name>.py` and one entry in [`registry.py`](registry.py):

```python
register_provider(
    DensityProvider(
        name="mymodel",                       # the value of Initial_Density
        title="MyModel",
        bridge=BRIDGES / "mymodel.py",
        models=("small", "large"),            # accepted ML_Density_Model values
        default_model="small",
        repository_variable="MYMODEL_REPO",   # discovery, as for the others
        python_variable="MYMODEL_PYTHON",
        find_checkpoint=lambda repository, model: repository / "weights" / f"{model}.pt",
        checkpoint_hint="download the weights from ...",
        source="https://github.com/...",
        summary="one line for ml_density.py models",
    )
)
```

`Initial_Density: mymodel` then works in `parsec.in`, in `ml_density.py`,
in the cache and in the GPU driver, and the workspace folder
`parsec_ml_workspace/mymodel/` is searched.

## 6. Reference

### `parsec.in` keys

| Key | Default | Meaning |
|---|---|---|
| `Initial_Density` | `sad` | `sad`, `file`, `charge3net`, `scdp`, `custom`, or a registered model |
| `ML_Density_Model` | `qm9` (ChargE3Net), `fast` (SCDP), `custom` | which trained model |
| `ML_Density_Repository` | discovered | the model's code (working directory of the bridge) |
| `ML_Density_Python` | discovered | the interpreter of the model's environment |
| `ML_Density_Checkpoint` | discovered | the trained weights |
| `ML_Density_Bridge` | — | the bridge script of `custom` |
| `ML_Density_Device` | `auto` | `auto` (GPU when PyTorch sees one), `cuda`, `cpu` |
| `ML_Density_Chunk_Size` | `50000` | points per model call (an upper bound) |
| `ML_Density_File` | — | import this prediction instead of running the model |
| `ML_Density_Units` | `auto` | units of a file without them (`e_per_bohr3`, `e_per_angstrom3`, `electrons_per_voxel`) |
| `ML_Density_Interpolation` | `linear` | for box files: `linear` or `nearest` |
| `ML_Density_Negative_Policy` | `clip` | `clip`, `error`, or `allow` negative values |
| `ML_Density_Cache` | `.parsec_ml_density_cache` next to `parsec.in` | cache folder |
| `ML_Density_Regenerate` | `false` | run the model even with a cached prediction |
| `Normalize_Initial_Density` | `true` | scale the start to the valence-electron count |

Discovery order for the repository: `ML_Density_Repository`, the environment
variable (`CHARGE3NET_REPO`, `SCDP_REPO`), `$PARSEC_ML_WORKSPACE/<model>`,
then `parsec_ml_workspace/<model>` next to the checkout. For the
interpreter: `ML_Density_Python`, `CHARGE3NET_PYTHON`/`SCDP_PYTHON`, then
`.venv` inside the repository.

### Command line

```text
python src/parsec_python/ml_density.py models
python src/parsec_python/ml_density.py check   [model options]
python src/parsec_python/ml_density.py predict (--input parsec.in [--pp-dir DIR] | --xyz FILE [--no-center])
                                               [--box L.. | --vacuum V..] [--points N.. | --spacing H..]
                                               [-o FILE.npz] [--cube FILE.cube] [--quiet] [model options]
model options: --provider NAME --model NAME --repository DIR --python EXE --checkpoint FILE
               --bridge FILE --device auto|cpu|cuda --chunk-size N
```

`python -m parsec_python.MLDensity ...` is the same when `src` is on the
Python path. With `--input`, model options not given are taken from the
input when it names the same model.

### File formats

| File | Content | Imported |
|---|---|---|
| point `.npz` (`predict --input`, the cache) | `density` (M,), `coordinates_bohr` (M, 3), `units` | exactly, after checking the points |
| box `.npz` (`predict --xyz`, other codes) | `density` (nx, ny, nz), `origin_bohr`, `voxel_vectors_bohr` (rows: steps of i, j, k), `units` | interpolated: `r(i,j,k) = origin + [i,j,k] @ voxel_vectors` |
| legacy `.npy` | 3-D array on exactly the calculation's Cartesian grid, e/Å³ | by index; do not reuse after changing the grid |
| `.cube` (`--cube`) | Gaussian cube, bohr and e/bohr³ | not imported; for viewers |

Files written by `predict` also record `atom_symbols` and
`atom_positions_bohr`. When a calculation has as many atoms and their mean
position differs by more than 0.25 bohr, the import stops: the density
belongs to another frame (an XYZ file in its own coordinates, say). A
slightly different geometry is accepted as a start. Files of other origin
need only `density`, `units`, and either `coordinates_bohr` or
`origin_bohr` + `voxel_vectors_bohr`; `save_point_density` and
`save_box_density` write them from Python.

### Cache

An entry `<model>-<16 hex digits>/` holds `request.npz`, `density.npz`, and
`manifest.json` with the full key: the bridge, model, interpreter, checkpoint
(path, size, time), the model repository's Git revision, the grid, and the
atoms. Any change runs the model again. A bridge writes `density.partial.npz`
and it is renamed only after a successful run, so an interrupted prediction
is never reused. Delete the folder to free the space.

### Scientific contract

The active PARSEC grid is authoritative: a model returns `rho(r_i)` at the
ordered grid points, in electrons per unit volume. PARSEC.py converts it to
e/bohr³, checks that it is finite, clips negative values (by default), and
normalizes its integral to the valence-electron count. The frozen core
density of the nonlinear core correction always comes from the
pseudopotential files. The models run with periodic boundaries disabled;
the box written in a request only encloses the points.

## 7. Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `... dependencies are missing in this Python environment`, `No module named torch` | the bridge ran in the wrong interpreter: set `ML_Density_Python` (or `--python`) to the model environment's `python` |
| `cannot locate the ... repository` | not in `parsec_ml_workspace` next to the checkout: give `ML_Density_Repository` or the environment variable |
| `no SCDP checkpoint for model ...` | unpack the Zenodo archive below `scdp/models/`, or give `ML_Density_Checkpoint` |
| `... uses virtual nodes 'none', but the checkpoint ... is a 'bond' model` | `ML_Density_Model` does not match the checkpoint; use `accurate` for `qm9_bond_...`, `fast` for `qm9_none_...` |
| `the SCDP checkpoint has no Gaussian basis for atomic number(s) ...` | an element outside QM9 (H, C, N, O, F) |
| `... was predicted for these atoms translated by ...` | the density file is in another frame: predict with `--input parsec.in`, or from the XYZ without `--no-center` |
| `An Application Control policy has blocked this file` (Windows) | Smart App Control blocks unsigned programs, e.g. interpreters downloaded by uv or DLLs of some wheels. Use python.org interpreters (`py -3.x -m venv`); if a package DLL is blocked, predict on another machine (section 4B) |
| `[WinError 206]` (path too long) | a deep folder on Windows: set `ML_Density_Cache` to a short path |
| CUDA out of memory, or very slow on a small GPU | ChargE3Net sizes its calls to half the free GPU memory; set `PARSEC_CHARGE3NET_MAX_PROBE_EDGES` lower, or `ML_Density_Device: cpu`. Free GPU memory matters: other processes on the GPU reduce it |
| `a spin-polarized calculation starts from the superposition of atomic densities` | expected: ML starts are for spin-restricted calculations |

The [42-case regression suite](../../../examples/ml_initial_density/README.md)
runs SAD, SCDP and ChargE3Net starts for 14 molecules from stored predictions.
