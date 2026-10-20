# Initial structures from machine-learned force fields

A DFT calculation of `parsec_python` starts from the coordinates of the
input. With `Initial_Structure: mlff` it first relaxes them with a
machine-learned interatomic potential, UMA (the default), MACE or any ASE
calculator, and starts from the relaxed structure instead. A relaxation,
molecular dynamics or a single point then begins close to the DFT minimum.

It pays for structures built by hand, taken from a crystal, or made by a
cheap method: on nanodiamonds expanded by 3 % it saved 37 to 49 % of a DFT
relaxation (numbers at the end). For a structure that is already relaxed it
changes little, and for a molecule of a few atoms the half a minute the force
field takes can exceed what it saves.

**Contents:** [Getting started](#getting-started) (Linux and macOS, Windows) ·
[On a cluster](#on-a-cluster) · [Using it](#using-it) · [Models](#models) ·
[Keywords](#keywords) · [What the run does](#what-the-run-does) ·
[Troubleshooting](#troubleshooting) · [What it gains, and which model](#what-it-gains-and-which-model) ·
[From Python](#from-python)

## Getting started

Four steps, done once per machine. The force field runs in a Python
environment of its own: it needs NumPy, ASE and the model package (PyTorch
comes with it), not `parsec_python`, so the model's PyTorch never mixes with
the CuPy of the DFT environment.

### Linux and macOS

```bash
# 1. An environment with the model (Python 3.10 or newer; 3.12 and 3.13 were used)
python3 -m venv ~/venvs/mlff
~/venvs/mlff/bin/pip install fairchem-core ase      # UMA; on Linux PyPI's PyTorch includes CUDA

# 2. UMA's weights are gated: accept the licence at https://huggingface.co/facebook/UMA
#    (log in there, fill the form), then log this machine in (choose "browser"):
~/venvs/mlff/bin/hf auth login

# 3. Tell PARSEC.py where the environment is (in ~/.bashrc, ~/.zshrc, ...)
echo 'export PARSEC_MLFF_PYTHON=$HOME/venvs/mlff/bin/python' >> ~/.bashrc
source ~/.bashrc

# 4. Check: downloads the weights (about 1.1 GB) and evaluates a water molecule
python src/parsec_python/mlff.py check
```

The check prints the model, the device and `water: E = -2079.86... eV`. On
macOS PyTorch has no CUDA and the model runs on the CPU (not tested here).
A conda environment works the same way (`conda create -n mlff python=3.12`,
then `pip install fairchem-core ase` inside it, and its `python` in step 3).

### Windows (PowerShell)

```powershell
# 1. An environment with the model, made with uv (installed into Python first).
#    PyPI's PyTorch for Windows is CPU-only: the extra index gives the CUDA build
#    for an NVIDIA GPU (driver 580 or newer).
py -3.12 -m pip install uv
py -3.12 -m uv venv $HOME\venvs\mlff --python 3.12
py -3.12 -m uv pip install --python $HOME\venvs\mlff\Scripts\python.exe fairchem-core ase `
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

# 2. Accept the licence at https://huggingface.co/facebook/UMA, then (choose "browser"):
& $HOME\venvs\mlff\Scripts\hf.exe auth login

# 3. Tell PARSEC.py where the environment is (new terminals see it)
setx PARSEC_MLFF_PYTHON "$HOME\venvs\mlff\Scripts\python.exe"

# 4. In a new terminal, from the repository root:
python src\parsec_python\mlff.py check
```

Without an NVIDIA GPU leave the extra index out; the model then runs on the
CPU. `fairchem-core` 2.23 needs PyTorch 2.13, whose CUDA builds for Windows
are on the `cu130` index; for a later version take the index that
<https://pytorch.org/get-started/locally/> names. No C++ compiler and no
Triton are needed: without Triton (which PyTorch for Windows lacks) UMA takes
its PyTorch kernels on the GPU. uv rather than pip, because on a Windows 11
laptop with Smart App Control on, the unsigned library of NVIDIA Warp that
fairchem loads was refused when pip had installed it
(`OSError: [WinError 4551]`) and accepted when uv had; a plain
`py -3.12 -m venv` + `pip install` works where that control is off. Loading
the model took about two minutes on that laptop (Windows scans the files on
first use), half a minute on Linux.

### Without a Hugging Face login: MACE

MACE models download without a login. Replace step 1 by

```bash
~/venvs/mlff/bin/pip install mace-torch ase          # Windows: add --extra-index-url https://download.pytorch.org/whl/cu128
```

skip step 2, and name the model in the input (`MLFF_Model: mace-mp-medium-mpa-0`)
or in the check (`mlff.py check mace-mp-medium-mpa-0`). It was less accurate
than UMA on the nanodiamonds (see the end), and the MACE-OFF and MACE-OMOL
models failed on large clusters.

## On a cluster

The same four steps, on a login node, with three things to watch:

- **Where the environment lives.** Importing PyTorch and the model from a
  slow shared file system can take minutes: on Perlmutter, from the project
  file system (CFS), fairchem took 40 to 110 s to import and UMA 100 to 250 s
  to load; from scratch, 5 s and 14 s. Put the environment on scratch or on
  the file system the centre recommends for software (Perlmutter:
  `/global/common/software/<project>`). Scratch may be purged after weeks
  without use; the environment is then made again.
- **Where the weights go.** They are cached in the home directory
  (`~/.cache/fairchem`, `~/.cache/mace`, about 1.1 GB per UMA model), which
  can fill a small home quota, and some compute nodes mount the home
  directory without file locks (Perlmutter: `OSError: [Errno 524]`). Point
  the caches to scratch and download the weights on the login node, where
  the network and the locks are there, by running step 4 there.
- **Which interpreter the job sees.** Export `PARSEC_MLFF_PYTHON` in the
  batch script (or in `~/.bashrc`), or write `MLFF_Python` into `parsec.in`.
- **pip's own cache** also goes to the home directory (several GB for
  PyTorch); `pip install --no-cache-dir` keeps it out.

On Perlmutter, for example (login node):

```bash
module load python
python -m venv $SCRATCH/venvs/mlff
$SCRATCH/venvs/mlff/bin/pip install --no-cache-dir fairchem-core ase
$SCRATCH/venvs/mlff/bin/hf auth login
cat >> ~/.bashrc <<'EOF'
export PARSEC_MLFF_PYTHON=$SCRATCH/venvs/mlff/bin/python
export FAIRCHEM_CACHE_DIR=$SCRATCH/mlff_cache/fairchem   # UMA weights
export XDG_CACHE_HOME=$SCRATCH/.cache                   # MACE weights (and other caches)
export HF_HOME=$HOME/.cache/huggingface                 # keeps the Hugging Face login where it is
EOF
source ~/.bashrc
python src/parsec_python/mlff.py check                  # downloads the weights here
```

and a batch script that relaxes with UMA first and then runs the DFT
relaxation on four GPUs (the launch script passes `--mlff` on to the
runner, whose first rank runs the force field):

```bash
#!/bin/bash
#SBATCH -A <account> -C gpu -q regular -N 1 --gpus-per-node=4 -t 01:00:00
export PARSEC_MLFF_PYTHON=$SCRATCH/venvs/mlff/bin/python
export FAIRCHEM_CACHE_DIR=$SCRATCH/mlff_cache/fairchem
export HF_HOME=$HOME/.cache/huggingface
scripts/run_multi_gpu_scf.sh calculation/parsec.in results/run1 --mlff
```

## Using it

Two lines in `parsec.in`:

```text
Initial_Structure: mlff
MLFF_Model: uma-s-1p2            # the default; or mace-mp-medium-mpa-0, my.model, ...
```

or, for an input without them, the command-line option:

```bash
python src/parsec_python/main.py calculation/parsec.in --mlff                 # the default model
python src/parsec_python/main.py calculation/parsec.in --mlff mace-mp-medium-mpa-0
```

`reference_main.py`, `scripts/run_multi_gpu_scf.sh` and the multi-node runner
take the same option. To look at the relaxed structure without a DFT run:

```bash
python src/parsec_python/mlff.py relax calculation/parsec.in
```

It writes `mlff_structure.xyz` beside the input; a DFT run with the same input
and settings takes that file instead of relaxing again, and `--restart-from
mlff_structure.xyz` starts any run from it. `examples/0d_mlff_initial_structure`
relaxes a benzene molecule built by hand and two nanodiamonds (C35H36,
C185H124) expanded by 3 %, with and without the force field.

## Models

`MLFF_Model` names one of them; `mlff.py models` lists them and whether an
interpreter has their package.

| Model | `MLFF_Model` | Install | Notes |
|---|---|---|---|
| UMA (Meta FAIR) | `uma-s-1p2` (default), `uma-s-1p2p1`, `uma-s-1p1`, `uma-m-1p1`; eSEN, AllScAIP names of fairchem | `pip install fairchem-core` | Weights gated: accept the licence at <https://huggingface.co/facebook/UMA>, then `hf auth login` (or `HF_TOKEN`). `MLFF_Task: omol` for molecules and clusters. A local checkpoint `*.pt` works without the hub. |
| MACE-MP | `mace-mp-<name>`, e.g. `mace-mp-medium-mpa-0` | `pip install mace-torch` | Materials (PBE); names of `mace_mp`. No login. |
| MACE-OFF23 | `mace-off-small`, `mace-off-medium`, `mace-off-large` | `pip install mace-torch` | Organic molecules (H C N O F P S Cl Br I); Academic Software License. Contracted large nanodiamonds. |
| MACE-OMOL-0 | `mace-omol` | `pip install mace-torch` | Molecules, trained on OMol25 like UMA's `omol` task; Academic Software License. Failed for clusters of a few hundred atoms. |
| Your own MACE | `path/to/file.model` | `pip install mace-torch` | Relative to the input. |
| Anything else | `python:module:function` or `python:path/file.py:function` | | The function is called with `device`, `task`, `charge`, `spin` and returns an ASE calculator. |

The model runs on the GPU when its PyTorch sees one (`MLFF_Device: auto`).
UMA runs without `torch.compile`, which costs more than a relaxation of some
tens of steps saves; `PARSEC_MLFF_COMPILE=on` compiles it where the C++
compiler of the platform is found (`g++`, or `cl` on Windows). A download of
weights that stalls for 300 s ends the run (`PARSEC_MLFF_NETWORK_TIMEOUT`).

## Keywords

| Keyword | Default | Meaning |
|---|---|---|
| `Initial_Structure` | `input` | `mlff` relaxes the input structure with the force field first. |
| `MLFF_Model` | `uma-s-1p2` | The potential, see the table above. |
| `MLFF_Task` | `omol` | UMA task (`omol`, `omat`, `oc20`, `odac`, `omc`). |
| `MLFF_Fmax` | 0.05 eV/ang | The relaxation stops when the largest force is below it; units as `Force_Min` (eV/ang, Ry/bohr, Hartree/bohr), eV/ang when none is given. |
| `MLFF_Steps` | 500 | Largest number of optimizer steps. |
| `MLFF_Optimizer` | `lbfgs` | ASE optimizer: `lbfgs`, `bfgs` or `fire`. |
| `MLFF_Device` | `auto` | `cuda`, `cpu`, or `auto` (CUDA when PyTorch sees a GPU). |
| `MLFF_Python` | | Interpreter that has the model package; a name on the `PATH` or a path relative to the input. |
| `MLFF_Charge` | `Net_Charges` | Charge in elementary charges, for models that use it (UMA `omol`, MACE-OMOL). |
| `MLFF_Spin` | 1 or 2 | Spin multiplicity 2S+1; by default the smallest one of the electron count. |
| `MLFF_Reuse` | `true` | Take `mlff_structure.xyz` again when it was relaxed from the same input with the same settings. |

## What the run does

1. The input structure is centred as the DFT run centres it.
2. The point group the DFT calculation will use is kept: forces and
   positions are averaged over the operations of the structure that respect
   `Move_Flag`, so that the run keeps its symmetry sectors. Atoms that
   `Move_Flag` fixes stay where they are. Molecular dynamics and
   `Ignore_Symmetry` impose no symmetry, as in the DFT run.
3. The ASE optimizer relaxes the structure to `MLFF_Fmax`.
4. The relaxed coordinates replace those of the input, as an XYZ file of
   `--restart-from` would; the domain stays the one of the input.

It writes beside the input (the runner: into its output directory):

| File | Content |
|---|---|
| `mlff_structure.xyz` | The relaxed structure in angstrom, with the model, energy and largest force; `--restart-from` accepts it. |
| `mlff_relaxation.xyz` | Every optimizer step with its energy (eV) and forces (eV/ang), extended XYZ. |
| `mlff.log` | The table of the ASE optimizer. |

and `parsec.out` reports the model, the number of steps, energy and largest
force before and after, the largest displacement and the time. The program
time at the end of `parsec.out` counts from after the force field.

The force field is not used when a restart file gives the structure
(`Relax_Restart`, `--restart-from`, `Restart_mode` of molecular dynamics) or
in a `manual` run, and the run says so. Periodic systems are refused.

## Troubleshooting

| Message or symptom | Cause | What to do |
|---|---|---|
| `fairchem.core cannot be imported by ...` (or `mace.calculators`) | The interpreter that runs the model lacks the package | Install it there, or point `PARSEC_MLFF_PYTHON` / `MLFF_Python` to the environment that has it; `mlff.py models` shows what an interpreter has. |
| `MLFF_Python ... cannot be started` | Wrong path | On Windows the interpreter is `...\Scripts\python.exe`, on Linux `.../bin/python`. |
| `GatedRepoError`, `401`, "UMA weights are gated" | No licence or no login | Accept the licence at <https://huggingface.co/facebook/UMA>, run `hf auth login` with the environment's `hf`; `hf auth whoami` shows the login. Setting `XDG_CACHE_HOME` moves the login too: set `HF_HOME=$HOME/.cache/huggingface` with it. |
| `OSError: [WinError 4551]` (Windows), on a library of `warp` | Smart App Control refused an unsigned library | Make the environment with uv (Windows steps above). |
| `No module named 'triton'` | A fairchem version that needs Triton on the GPU | `MLFF_Device: cpu`, or `pip install triton-windows` on Windows. |
| `OSError: [Errno 524]` | The cache directory has no file locks (compute node) | Run `mlff.py check` on a login node, or set `FAIRCHEM_CACHE_DIR` / `XDG_CACHE_HOME` to scratch. |
| `Disk quota exceeded` | The weights or pip's cache filled the home directory | Move `~/.cache/fairchem`, `~/.cache/mace` to scratch and set the variables above; install with `pip install --no-cache-dir`. |
| The first run takes minutes | Downloads, or imports from a slow file system | Run `mlff.py check` once; keep the environment on a fast file system. |
| `does not describe this structure` | The model gave forces above 50 eV/ang, or its energy rose | Another model (`uma-s-1p2`, `mace-mp-medium-mpa-0`); the steps are in `mlff_relaxation.xyz`. |
| `moved an atom out of the domain` | The relaxed structure does not fit the sphere of the input | Check `mlff_relaxation.xyz`; a larger `Boundary_Sphere_Radius`, or another model. |
| `CUDA out of memory` in the force field | Model too large for the GPU (`uma-m-1p1` at 1095 atoms) | A small model, or `MLFF_Device: cpu`. |
| `the force field ended with exit status ...` | Any other error of the model | The last lines of its error follow the message. |

## What it gains, and which model

Measured on the hydrogen-terminated nanodiamonds of `examples/0d_nanodiamonds`
(CA-LDA, BFGS to `Force_Min` 0.001 Ry/bohr with the SCF criterion 1e-6 Ry,
A100 GPUs, the model environments on scratch). The structures of the example
are GFN2-xTB minima; a crude start was made by expanding each by 3 %.

How close each model brings the expanded structure to the DFT minimum, RMSD
in angstrom, and its time for the whole relaxation, model loading included:

| Model | C35H36 | C185H124 | C795H300 | C1213H412 | Time |
|---|---:|---:|---:|---:|---:|
| `uma-s-1p2` (default) | 0.052 | 0.044 | 0.023 | 0.022 | 20 to 29 s |
| `uma-s-1p2p1` | 0.053 | 0.050 | 0.035 | 0.032 | 19 to 29 s |
| `uma-s-1p1` | 0.053 | 0.052 | 0.028 | 0.095 | 38 to 111 s |
| `uma-m-1p1` | 0.052 | 0.047 | out of memory | out of memory | |
| `mace-mp-medium-mpa-0` | 0.059 | 0.068 | 0.046 | 0.051 | 29 to 54 s |
| `mace-off-medium` | 0.051 | 0.141 | 0.326 | 0.392 | |
| `mace-omol` | 0.052 | refused | refused | out of memory | |
| (the xTB structure) | 0.045 | 0.038 | 0.023 | 0.028 | |
| (the expanded structure) | 0.129 | 0.190 | 0.257 | 0.284 | |

DFT structures to the criterion and the time on 4, 4, 8 and 16 GPUs (the
final energies of a cluster agree within 0.4 meV/atom):

| Cluster | xTB structure | expanded | expanded, then `uma-s-1p2` | time: expanded / `uma-s-1p2` + DFT |
|---|---:|---:|---:|---|
| C35H36 | 13 | 29 | 13 | 33 s / 30 + 22 s |
| C185H124 | 21 | 38 | 18 | 122 s / 19 + 57 s |
| C795H300 | 17 | 27 | 20 | 249 s / 35 + 114 s |
| C1213H412 | 22 | 31 | 16 | 291 s / 26 + 124 s |

- From a crude structure, `uma-s-1p2` takes the DFT run back to the number
  of steps of the xTB structure or fewer; for the larger clusters it saves
  37 to 49 % of the time. From a structure that is already good (the xTB
  minima) a force field changes little.
- The model matters. `mace-off-medium` contracted C795H300 by about 3.5 % and
  formed 1.33 angstrom C-C bonds at its surface, and the DFT run took more
  steps than without it (34 against 27); `mace-omol` gave forces of hundreds
  of eV/angstrom for C185H124 and larger; `uma-m-1p1` needed more than the
  80 GB of an A100 for C795H300.
- Without compilation and with the environment on a fast file system the
  force field costs some tens of seconds; on a project file system with
  `torch.compile`, several minutes.

A structure that a model does not describe is refused before the DFT run:
forces above 50 eV/angstrom or not finite, an energy that rises, or an atom
pushed out of the domain stop the run with a message that names other
models; the trajectory stays in `mlff_relaxation.xyz` for inspection.

## From Python

```python
from parsec_python.Input import parse_parsec_input
from parsec_python.MLFF import relax_initial_structure, apply_mlff_option

translation = apply_mlff_option(parse_parsec_input("calculation/parsec.in"), "uma-s-1p2")
translation, notes = relax_initial_structure(translation, "calculation")
print("\n".join(notes))
```
