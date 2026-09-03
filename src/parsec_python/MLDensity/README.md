# ML initial-density interface

`MLDensity` makes a machine-learned valence density an optional SCF initial
guess. It does not replace any DFT physics: the PARSEC grid, pseudopotentials,
Hartree/XC construction, Hamiltonian, eigensolver, mixing, total energy, and
convergence test remain unchanged.

## Start here: one calculation from installation to `parsec.out`

The simplest workflow does **not** require the user to run a model script or
construct a density grid manually. PARSEC.py launches the selected model in
its own Python environment, asks it to predict directly on the active PARSEC
grid, saves the prediction as a portable `.npz`, and then uses that density as
the first SCF guess in the same command.

The examples below use Windows PowerShell. On Linux, replace
`Scripts\python.exe` with `bin/python` and use ordinary POSIX paths.

### 1. Install PARSEC.py

Clone PARSEC.py and create its DFT environment. This environment does not need
PyTorch, ChargE3Net, or SCDP:

```powershell
cd C:\work
git clone https://github.com/QianGroupPage/PARSEC.py.git parsec_python
cd parsec_python
py -3.12 -m venv .venv312
& ".\.venv312\Scripts\python.exe" -m pip install `
  -r .\src\parsec_python\requirements.txt
```

Prepare an ordinary isolated PARSEC calculation. All pseudopotentials used by
`parsec.in` must be present in the calculation directory (or supplied with
`--pp-dir`):

```text
C:\work\calculation\
├── parsec.in
├── C_POTRE.DAT
└── H_POTRE.DAT
```

Start with an input known to run with `Initial_Density: sad`. The sections
below show the small block that changes it to an ML initial density.

### 2A. ChargE3Net: install, predict, and run

Clone the official ChargE3Net repository beside PARSEC.py and give it a
separate environment:

```powershell
cd C:\work
git clone https://github.com/AIforGreatGood/charge3net.git
py -3.12 -m venv .\charge3net\.venv
& ".\charge3net\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\charge3net\.venv\Scripts\python.exe" -m pip install `
  -r ".\charge3net\requirements.txt"
```

Install a CUDA-enabled PyTorch build in that environment if GPU inference is
desired. Confirm that the QM9 checkpoint distributed with the repository is
present:

```powershell
Test-Path C:\work\charge3net\models\charge3net_qm9.pt
& "C:\work\charge3net\.venv\Scripts\python.exe" -c `
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For an isolated molecule containing only H, C, N, O, and F, replace the SAD
line in `C:\work\calculation\parsec.in` with:

```text
Initial_Density: charge3net
ML_Density_Model: qm9
ML_Density_Repository: C:\work\charge3net
ML_Density_Python: C:\work\charge3net\.venv\Scripts\python.exe
ML_Density_Device: auto
ML_Density_Cache: .parsec_ml_density_cache
Normalize_Initial_Density: true
```

Run PARSEC.py from the PARSEC.py repository:

```powershell
cd C:\work\parsec_python
& ".\.venv312\Scripts\python.exe" ".\src\parsec_python\main.py" `
  "C:\work\calculation\parsec.in" --no-archive
```

This single command performs the following operations:

1. reads the atoms and real-space grid from `parsec.in`;
2. launches the ChargE3Net environment;
3. evaluates the QM9 checkpoint at the exact active grid points;
4. writes a cached `density.npz`;
5. validates, clips, and normalizes that density;
6. runs the ordinary SCF calculation and writes `parsec.out`.

The generated density can be located with:

```powershell
Get-ChildItem "C:\work\calculation\.parsec_ml_density_cache" `
  -Recurse -Filter density.npz
```

### 2B. SCDP: install, predict, and run

SCDP's published dependencies require their own environment. Python 3.10 is
the most straightforward choice for its pinned PyTorch 1.13.1/CUDA 11.6
stack:

```powershell
cd C:\work
git clone https://github.com/kyonofx/scdp.git
py -3.10 -m venv .\scdp\.venv
& ".\scdp\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\scdp\.venv\Scripts\python.exe" -m pip install `
  -r ".\scdp\requirements.txt"
& ".\scdp\.venv\Scripts\python.exe" -m pip install -e ".\scdp"
```

The official SCDP repository does not contain the pretrained weights. Download
either the `fast` or `accurate` QM9 checkpoint from the Zenodo link in the
[SCDP README](https://github.com/kyonofx/scdp#pretrained-models). In this
example it is saved as `C:\work\checkpoints\scdp_fast.ckpt`.

Replace the SAD line in `parsec.in` with:

```text
Initial_Density: scdp
ML_Density_Model: fast
ML_Density_Repository: C:\work\scdp
ML_Density_Python: C:\work\scdp\.venv\Scripts\python.exe
ML_Density_Checkpoint: C:\work\checkpoints\scdp_fast.ckpt
ML_Density_Device: auto
ML_Density_Cache: .parsec_ml_density_cache
Normalize_Initial_Density: true
```

Run the same PARSEC.py command:

```powershell
cd C:\work\parsec_python
& ".\.venv312\Scripts\python.exe" ".\src\parsec_python\main.py" `
  "C:\work\calculation\parsec.in" --no-archive
```

SCDP's prediction is written under a directory beginning with `scdp-` inside
the same `.parsec_ml_density_cache`. The final calculation output remains
`C:\work\calculation\parsec.out`.

### 3. Reuse the generated `.npz` without either ML installation

The cache key contains the model, checkpoint, geometry, and exact PARSEC grid.
For a permanent, self-explanatory copy, copy the newest result into the
calculation directory. For example, after a ChargE3Net run:

```powershell
$predictedDensity = Get-ChildItem `
  "C:\work\calculation\.parsec_ml_density_cache\charge3net-*\density.npz" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
Copy-Item -LiteralPath $predictedDensity.FullName `
  -Destination "C:\work\calculation\charge3net_qm9_density.npz"
```

The calculation can now be run on another machine without the ChargE3Net
repository, checkpoint, or PyTorch environment. Use this input block:

```text
Initial_Density: charge3net
ML_Density_File: charge3net_qm9_density.npz
ML_Density_Model: qm9
ML_Density_Units: auto
ML_Density_Interpolation: linear
ML_Density_Negative_Policy: clip
Normalize_Initial_Density: true
```

For SCDP, use `Initial_Density: scdp`, the copied SCDP `.npz`, and the matching
`ML_Density_Model`. When `ML_Density_File` is supplied, all repository,
checkpoint, Python, device, and cache options are optional and no external ML
process is launched.

### Which density file format should a new user use?

Use `.npz` whenever possible. The direct adapters write a one-dimensional
density together with the exact ordered PARSEC coordinates, units, provider,
model, and checkpoint provenance. It is therefore unambiguous and safe to
reuse for the same physical grid.

A plain `.npy` is supported mainly for old-architecture compatibility. It must
contain a three-dimensional volume-density array whose shape exactly matches
the full underlying Cartesian PARSEC grid. With `ML_Density_Units: auto`, it is
interpreted as electrons per cubic angstrom. Because `.npy` contains neither
coordinates nor voxel vectors, do not reuse one after changing grid spacing,
domain radius, grid shift, or domain shape.

For an externally generated file with a different structured grid, create the
documented `.npz` form containing `density`, `origin_bohr`,
`voxel_vectors_bohr`, and `units`; PARSEC.py can then interpolate it onto the
active grid. The direct adapters avoid this interpolation by predicting at the
active points from the beginning.

## Scientific contract

The active PARSEC grid is authoritative. A provider must return the volume
density `rho(r_i)` at those ordered points, in electrons per unit volume.
PARSEC.py then:

1. converts the field to `e/bohr^3`;
2. validates coordinates and finite values;
3. clips small negative predictions by default;
4. normalizes the integral to the pseudopotential valence-electron count;
5. uses the result only as the first SCF density.

The frozen nonlinear-core-correction density always comes from
`POTRE.DAT`. A core-hole label such as `C-1s` is presented to the model as its
chemical element `C`; a conventional density model does not encode the
core-hole configuration, so this remains an approximate starting guess. A
fully converged SCF result is independent of a sufficiently reasonable
initial guess.

## Does a model need XYZ, POSCAR, or a cell?

Both models need atomic species and Cartesian positions. Density prediction
also needs the coordinates at which the density is requested. Periodic model
workflows additionally need a physical cell and periodic-boundary flags, so a
bare XYZ is not generally sufficient.

For the isolated PARSEC.py interface, no separate XYZ/POSCAR is needed. The
adapter reads atoms from `parsec.in`, evaluates the model at the exact active
DFT grid points, constructs a containing orthorhombic cell from the underlying
Cartesian grid, and disables PBC. Atoms and probes are translated together,
which cannot affect these translation-equivariant molecular predictions.

## Configuration reference

### Dependency isolation and automatic path discovery

ChargE3Net and SCDP use different PyTorch/e3nn generations. Keep each in its
own environment; the DFT environment launches a bridge process and exchanges
only portable `.npz` files. Neither ML stack is imported for an
`Initial_Density: sad` or file-based calculation.

The explicit `ML_Density_Repository` and `ML_Density_Python` entries shown in
the quick start are easiest to understand and reproduce. They may be omitted
when the following recommended sibling layout is used:

```text
parsec_ml_workspace/
├── parsec_python/                   junction or checkout of this repository
├── parsec_python_old_architecture/  optional old-architecture worktree
├── charge3net/                      official clone + its .venv
└── scdp/                            official clone + its .venv
```

This layout does not change any Git remote, branch, or GitHub pointer. Custom
locations can also be supplied through `PARSEC_ML_WORKSPACE`,
`CHARGE3NET_REPO`, `CHARGE3NET_PYTHON`, `SCDP_REPO`, and `SCDP_PYTHON`.

### Model and checkpoint choices

| Provider | `ML_Density_Model` | Intended domain | Checkpoint behavior |
|---|---|---|---|
| ChargE3Net | `qm9` | Isolated H/C/N/O/F molecules | Automatically uses `models/charge3net_qm9.pt` |
| ChargE3Net | `mp` | Periodic Materials Project systems | Automatically uses `models/charge3net_mp.pt`; validate carefully before applying it to an isolated cluster |
| ChargE3Net | `nmc` | Periodic NMC cathode systems | Automatically uses `models/charge3net_nmc.pt` |
| SCDP | `fast` | Isolated QM9 molecules | Requires an explicit downloaded checkpoint; 4 layers and no virtual nodes |
| SCDP | `accurate` | Isolated QM9 molecules | Requires an explicit downloaded checkpoint; 8 layers and bond virtual nodes |

An explicit `ML_Density_Checkpoint` overrides ChargE3Net's conventional
filename. SCDP weights are distributed separately, so its checkpoint should
normally always be specified.

### Cache controls

The default cache is `.parsec_ml_density_cache` beside `parsec.in`. Its key
includes the provider, model, checkpoint, model code revision, atoms, and
exact grid. An unchanged calculation safely reuses the cached prediction.

Use `ML_Density_Regenerate: true` to rerun inference even if an exact cache
entry exists. Use `ML_Density_Cache: path` to place the cache elsewhere. A
previously copied `ML_Density_File` always takes precedence and requires no
cache or model process.

The offline file workflow is exercised by the complete
[`examples/ml_initial_density`](../../../examples/ml_initial_density/README.md)
regression suite.

### Portable file schema

The preferred `.npz` is either:

- one-dimensional `density`, matching ordered `coordinates_bohr`, and scalar
  `units`; or
- three-dimensional `density`, `origin_bohr`, a 3x3
  `voxel_vectors_bohr`, and `units`, enabling linear/nearest interpolation.

Supported units are `e_per_bohr3`, `e_per_angstrom3`, and
`electrons_per_voxel`. The Python helper `save_point_density(...)` writes the
exact-point format.

## Applicability warning

An adapter being able to evaluate a geometry does not establish model
transferability. Check that every element and bonding environment lies in the
checkpoint's training domain, compare integrated/raw densities, and benchmark
SCF iteration count and final energy against SAD. PARSEC.py deliberately
normalizes model charge but does not conceal an unsupported chemical domain.
