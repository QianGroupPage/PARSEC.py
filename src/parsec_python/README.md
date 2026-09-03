# `parsec_python` package

`parsec_python` is the canonical implementation of PARSEC-style isolated
real-space DFT in this repository. The package contains both readable
scientific components and accuracy-audited accelerated backends; users do not
need to choose between separate source trees.

The implementation is native Python/C++/CUDA. It does not launch the PARSEC
executable or call Fortran. PARSEC source and output are used as the algorithm
specification and numerical reference.

## Entry points

From the repository root:

```powershell
python src\parsec_python\main.py calculation\parsec.in --no-archive
```

From this directory:

```powershell
python main.py calculation\parsec.in --no-archive
```

As a package:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m parsec_python calculation\parsec.in --no-archive
```

All three commands select the optimized workflow. `--backend auto` combines
the fastest compatible SciPy, native C++/OpenMP, and CuPy components.
`--symmetry auto` detects exact supported operations and applies
representation decomposition when profitable. Missing optional acceleration
causes a reported safe fallback.

Default output files beside `parsec.in` are:

- `parsec.out`, containing PARSEC-shaped setup, SCF, energy, convergence, and
  timing sections;
- `parsec_python_results.npz`, containing structured arrays and metadata,
  unless `--no-archive` is supplied.

Use `reference_main.py` only when deliberately auditing the readable SciPy
translation:

```powershell
python src\parsec_python\reference_main.py calculation\parsec.in --no-archive
```

## Package organization

| Location | Responsibility |
|---|---|
| `Input/` | Translate and validate PARSEC/ESDF input. |
| `MLDensity/` | Load or predict optional SCDP/ChargE3Net initial densities on the exact DFT grid. |
| `Grid/` | Build centered isolated sphere or box grids. |
| `Laplacian/` | Generate high-order finite-difference coefficients and sparse `-nabla^2`. |
| `Pseudopotential/` | Read Martins-new `POTRE.DAT`, integrate radial data, and reproduce PARSEC splines. |
| `V_ion/` | Assemble local ionic fields, KB nonlocal factors, initial atomic density, and ion-ion energy. |
| `Hartree/` | Construct open-boundary data and solve the finite-difference Poisson equation. |
| `V_xc/` | Evaluate CA/PZ LDA or spin-unpolarized PBE and their energy terms. |
| `Hamiltonian/` | Apply the matrix-free Kohn--Sham operator. |
| `Eigensolvers/` | Readable CHEBFF, CHEBDAV, subspace, filtering, orthogonalization, and Ritz primitives. |
| `Occupations/` | Determine Fermi occupations and rebuild the density. |
| `Mixer/` | Apply PARSEC-style potential mixing and residual tests. |
| `Energy/` | Assemble the total-energy decomposition. |
| `SCF/` | Orchestrate one complete self-consistent single point. |
| `Output/` | Write PARSEC-shaped text and machine-readable results. |
| `acceleration/` | Implement optimized backends, GPU eigensolvers, symmetry sectors, resident execution, and the native extension. |
| `driver.py` | Prepare and run the readable reference workflow. |
| `acceleration/driver.py` | Prepare and run the default optimized workflow. |

The top-level package API intentionally exposes the optimized workflow:

```python
from parsec_python import prepare_single_point, run_scf, run_single_point
```

Explicit reference aliases expose the readable workflow without ambiguous
imports:

```python
from parsec_python import (
    prepare_reference_single_point,
    run_reference_scf,
    run_reference_single_point,
)
```

## Modular calculations

Every major physical stage is independently importable. For example:

```python
from parsec_python import (
    build_cluster_grid,
    build_local_ionic_potential,
    build_negative_laplacian,
    build_nonlocal_projectors,
    ca_lda,
    pbe,
    read_parsec_pseudopotential,
    solve_hartree,
)
```

`prepare_single_point(...)` constructs the grid and all static Hamiltonian
data but does not perform SCF. `run_scf(prepared)` consumes that prepared
system. This separation makes it possible to profile or validate grid,
Laplacian, ionic, Hartree, XC, and eigensolver components individually.

## Physical scope

Currently supported:

- isolated spherical and box domains;
- high-order Cartesian finite differences;
- scalar norm-conserving Martins-new pseudopotentials through `l=3`;
- separable Kleinman--Bylander nonlocal projectors;
- optional nonlinear core correction;
- CA/PZ LDA and spin-unpolarized PBE;
- potential-mixed SCF with PARSEC-style occupations and eigensolver policy;
- exact automatic reflection/signed-permutation representation reduction
  where supported;
- core-hole species labels such as `C-1s` with an explicit
  `Element_Symbol: C` and optional `Atomic_Energy_Correction`.

Not currently supported as production physics:

- periodic boundary conditions;
- spin polarization or spin--orbit coupling;
- forces and geometry relaxation;
- hybrid/meta-GGA functionals and DFT+U;
- PARSEC restart-file compatibility;
- Ono--Hirose double-grid order greater than one.

Unsupported requested input is rejected rather than silently approximated.
Sphere Hartree boundaries use a multipole expansion; box boundaries use the
exact, slower direct Coulomb construction.

Machine-learned densities are optional initial guesses, not new functionals.
They are converted/validated/normalized on the authoritative PARSEC grid and
do not replace the PP core density or any converged DFT term. See
[MLDensity/README.md](MLDensity/README.md) for direct SCDP/ChargE3Net setup,
input labels, cache behavior, and training-domain limitations.

## Pseudopotentials

The pseudopotential filename is normally `<Atom_Type>_POTRE.DAT`. A
configuration-qualified core-hole type therefore uses, for example,
`C-1s_POTRE.DAT` while `Element_Symbol: C` preserves chemical identity.

Pseudopotentials and the selected XC functional must be consistent. The
solver can read converted norm-conserving UPF data, but conversion does not by
itself validate transferability, ghost states, relativistic content, or
core-hole suitability.

## Examples and validation

Runnable calculations and benchmark data live in the repository-level
[`examples/`](../../examples/README.md) directory, beside `src/` rather than
inside this importable package. It includes compact parity cases, larger
performance systems such as naphthalene and `Si28H36`, and the specialized
core-hole PBE study. Local result files and caches can remain inside a case
directory but are not part of the maintained source interface.

Run the readable suite:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s src\parsec_python\tests -p "test_*.py" -v
```

Run backend, native, CUDA, symmetry, and parity tests:

```powershell
python -m unittest discover `
  -s src\parsec_python\acceleration\tests -p "test_*.py" -v
```

CUDA-specific tests execute only when CuPy and a working CUDA runtime are
present.

## Further reading

- [PARSEC_ALGORITHM.md](PARSEC_ALGORITHM.md): reviewed Fortran call path,
  formulas, defaults, and convergence policy.
- [PYTHON_IMPLEMENTATION.md](PYTHON_IMPLEMENTATION.md): Python mapping and
  parity status for each physical stage.
- [ARCHITECTURE.md](ARCHITECTURE.md): package boundaries and native-port rules.
- [acceleration/README.md](acceleration/README.md): backend selection,
  optimized kernels, resident execution, and profiling.
- [acceleration/ACCELERATION_AUDIT.md](acceleration/ACCELERATION_AUDIT.md):
  physics-preservation and performance audit.
