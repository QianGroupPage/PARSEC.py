# PARSEC.py

PARSEC.py contains native Python implementations of real-space density
functional theory (DFT), and supports machine-learned charge densities as an
alternative to the superposition of atomic densities (SAD) for initializing
SCF iterations. It accompanies the article below, accepted in the *Journal of
Computational Chemistry*. See [Citing PARSEC.py](#citing-parsecpy).

The repository currently keeps two independent code paths:

- `src/new_architecture`: the recommended, modular translation of PARSEC's
  isolated single-point workflow.
- `src/old_architecture`: the earlier Python implementation, retained for
  comparison, its broader experimental workflows, and CPU/GPU performance
  work.

The new implementation reads PARSEC-style `parsec.in` files and Martins-new
`*_POTRE.DAT` pseudopotentials directly. It does not call the PARSEC
executable, launch WSL or MPI, or bind to the Fortran code at runtime. PARSEC's
Fortran source is used as the algorithm specification and reference result.

## Contents

- [Quick start: new architecture](#quick-start-new-architecture)
- [Repository structure](#repository-structure)
- [Which implementation should I use?](#which-implementation-should-i-use)
- [New-architecture component map](#new-architecture-component-map)
- [Single-point calculation flow](#single-point-calculation-flow)
- [Modular Python API](#modular-python-api)
- [Supported physical scope](#supported-physical-scope)
- [Benchmarks](#benchmarks)
- [Testing the new architecture](#testing-the-new-architecture)
- [Old architecture](#old-architecture)
- [UPF-to-PARSEC conversion tool](#upf-to-parsec-conversion-tool)
- [Further documentation](#further-documentation)
- [Citing PARSEC.py](#citing-parsecpy)

## Quick start: new architecture

Python 3.10 or newer is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

This is a pure-Python install; no compiler is required. It pulls in NumPy and
SciPy, puts `new_architecture` and `old_architecture` on the import path, and
provides the `parsec-py` command. Add the legacy plotting dependencies with
`python -m pip install -e ".[legacy]"`.

Every later command uses `python` and forward-slash paths, which work in bash,
zsh, PowerShell, and `cmd` once the environment is active.

Put `parsec.in` and one pseudopotential per atom type in the same calculation
directory:

```text
H2/
├── parsec.in
└── H_POTRE.DAT
```

The filename must match the corresponding `Atom_Type`; for example, atom type
`H` resolves to `H_POTRE.DAT`.

Validate the input and POTRE files without constructing the real-space grid:

```bash
parsec-py path/to/parsec.in --dry-run
```

Run the physical calculation:

```bash
parsec-py path/to/parsec.in
```

For a small physical example, run the canonical H2 case from its benchmark
directory:

```bash
cd src/new_architecture/benchmarks/h2_canonical_nodg
parsec-py parsec.in --no-archive
```

It converges to `-2.13713415 Ry`. Without installing, the equivalent
folder-local launcher is `python src/new_architecture/main.py path/to/parsec.in`.

The default outputs are written beside `parsec.in`:

- `parsec.out`: PARSEC-shaped setup, SCF, energy, convergence, and timing
  report.
- `parsec_python_results.npz`: structured arrays and metadata, unless
  `--no-archive` is supplied.

Useful command-line controls are:

| Option | Purpose |
|---|---|
| `--dry-run` | Parse and validate the input and pseudopotentials without building the grid. |
| `--pp-dir DIR` | Search this directory first for `*_POTRE.DAT` files, before the input and current directories. |
| `--log FILE` | Change the text-output filename. |
| `--output FILE` | Change the NumPy archive filename. |
| `--no-archive` | Write only the text report. |
| `--save-wavefunctions` | Include wavefunctions in the result archive. |
| `--quiet` | Suppress console progress while retaining file output. |
| `--debug` | Include a traceback when a calculation fails. |

A converged calculation returns exit code 0. Other runtime failures return 1,
input errors return 2, and interruption returns 130. A valid calculation that
reaches `Max_Iter` without converging writes its final state and returns 3.

Package execution is also supported:

```bash
python -m new_architecture path/to/parsec.in
```

## Repository structure

```text
parsec_python/
├── src/
│   ├── new_architecture/       Modular PARSEC-style isolated single points
│   │   ├── Input/              PARSEC/ESDF input translation
│   │   ├── Grid/               Isolated real-space domains
│   │   ├── Laplacian/          Finite-difference kinetic operator
│   │   ├── Pseudopotential/    Martins-new POTRE reader and radial splines
│   │   ├── V_ion/              Local and nonlocal ionic terms
│   │   ├── Hartree/            Isolated Poisson boundary and CG solve
│   │   ├── V_xc/               CA/PZ LDA exchange-correlation
│   │   ├── Hamiltonian/        Kohn-Sham operator composition
│   │   ├── Eigensolvers/       CHEBFF, CHEBDAV, and SUBSPACE filtering
│   │   ├── Occupations/        Fermi filling and density construction
│   │   ├── Mixer/              Potential residuals and Anderson mixing
│   │   ├── Energy/             Total-energy bookkeeping
│   │   ├── SCF/                Static preparation and nonlinear SCF loop
│   │   ├── Output/             PARSEC-style reporting
│   │   ├── benchmarks/         H2, benzene, and naphthalene cases
│   │   ├── provenance/         Fortran-to-Python source map
│   │   ├── tests/              Unit, architecture, and integration tests
│   │   ├── models.py           Typed settings and result objects
│   │   ├── driver.py           Public prepare/run composition
│   │   ├── cli.py              Command-line and archive handling
│   │   └── main.py             Folder-local launcher
│   ├── old_architecture/       Preserved earlier Python implementation
│   │   ├── Eigensolvers/       CPU/GPU legacy eigensolver routines
│   │   ├── Laplacian/          Legacy finite-difference operators
│   │   ├── Mixer/              Legacy simple/multisecant mixing
│   │   ├── Splines/            Legacy spline utilities
│   │   ├── V_ion/              Legacy pseudopotential implementations
│   │   ├── V_xc/               Legacy CA-LDA CPU/GPU routines
│   │   ├── native/             C++/OpenMP ionic setup kernels (separate
│   │   │                       optional `rsdft-native` distribution)
│   │   ├── GUI/                Legacy input generator
│   │   ├── Tools/              Legacy timing/plot helpers
│   │   ├── main.py             Original monolithic script
│   │   └── main_new.py         Refactored legacy launcher
│   └── parsec_tools/           Architecture-independent conversion utilities
│       └── upf_to_parsec.py    UPF v2 to Martins-new POTRE converter
├── samples/                    Inputs/results for the older workflow
├── pyproject.toml              Pure-Python packaging for `parsec-py`
└── README.md
```

The two architecture folders do not call one another. This keeps reference
comparisons meaningful and allows the modular code to evolve without silently
falling back to an older algorithm.

## Which implementation should I use?

| Question | `new_architecture` | `old_architecture` |
|---|---|---|
| Primary purpose | Reproduce and inspect PARSEC isolated single-point algorithms | Preserve the earlier Python workflows and performance experiments |
| Main input | PARSEC/ESDF `parsec.in` | Manual input or `.in`, `.inp`, `.json`, `.dat`, `.mat`, `.txt` |
| Pseudopotentials | PARSEC Martins-new `*_POTRE.DAT` | `elements_new.csv`, `splineData.mat`, and legacy helpers |
| Organization | One package per physical concept | Monolithic driver plus partially refactored modules |
| Eigensolver path | Native Python CHEBFF or CHEBDAV, then saved SUBSPACE filtering | Several legacy CPU/GPU solver choices |
| Hardware path | CPU/SciPy; no MPI, GPU, or symmetry reduction | CPU, optional CuPy GPU, optional C++/OpenMP ionic setup |
| Recommended use | PARSEC parity work and modular component studies | Historical comparisons and older specialized workflows |

## New-architecture component map

The numerical modules expose independent APIs so a calculation stage can be
tested or reused without running a complete SCF job.

| Package or file | Principal API | Responsibility |
|---|---|---|
| `models.py` | `Atom`, `GridSettings`, `HartreeSettings`, `EigensolverSettings`, `SCFSettings`, `SinglePointInput`, `SinglePointResult` | Typed calculation inputs, settings, per-iteration data, and final results. |
| `Input/parsec_input.py` | `parse_parsec_input()`, `summarize_translation()` | Parse supported PARSEC/ESDF syntax, resolve required POTRE paths, apply units/defaults, and reject unsupported physics. |
| `Grid/cluster.py` | `RealSpaceGrid`, `build_cluster_grid()` | Build the full active sphere or box lattice, integer-to-row lookup, Cartesian coordinates, and integration weights. |
| `Laplacian/finite_difference.py` | `second_derivative_coefficients()`, `build_negative_laplacian()`, `apply_negative_laplacian_boundary()` | Build centered Fornberg stencils and sparse `-∇²`; fold known exterior Dirichlet values into a Poisson right-hand side. |
| `Pseudopotential/` | `read_parsec_pseudopotential()`, `ParsecRadialSpline`, `parsec_radial_integral()` | Read Martins-new POTRE channels, wavefunctions, occupations, valence/core densities, and perform PARSEC-style radial interpolation/quadrature. |
| `V_ion/ionic_potential.py` | `build_local_ionic_potential()`, `build_nonlocal_projectors()`, `superpose_atomic_density()`, `ion_ion_energy()` | Construct the local ionic field, normalized Kleinman-Bylander projectors, SAD/NLCC densities, and ion-ion repulsion. |
| `Hartree/poisson.py` | `density_multipoles()`, `MultipoleExpansion`, `DirectCoulombBoundary`, `solve_hartree()` | Set isolated Dirichlet boundaries and solve `(-∇²)V_H = 8πρ` with conjugate gradients. |
| `V_xc/ca_lda.py` | `ca_lda()` | Evaluate spin-unpolarized Ceperley-Alder/Perdew-Zunger LDA potential and energy, including optional frozen NLCC density. |
| `Hamiltonian/operator.py` | `KohnShamHamiltonian` | Compose kinetic, diagonal local, and low-rank nonlocal actions without forming a dense Hamiltonian. |
| `Eigensolvers/` | `solve_eigval()`, `run_chebff()`, `run_chebdav()`, `run_subspace_filter()` | Translate PARSEC spectral bounds, Chebyshev filtering, orthogonalization, Rayleigh-Ritz, locking/restart, and saved-subspace policy. |
| `Occupations/fermi_dirac.py` | `fermi_occupations()`, `density_from_orbitals()` | Determine the Fermi level, enforce the electron count, and form the spin-degenerate real-space density. |
| `Mixer/anderson.py` | `potential_residual_metrics()`, `AndersonMixer` | Compute plain/charge-weighted SRE and produce the next effective potential. |
| `Energy/total_energy.py` | `total_energy()` | Assemble PARSEC's input-Hxc/new-density double-counting expression and named energy components. |
| `SCF/single_point.py` | `PreparedSinglePointSystem`, `prepare_single_point()`, `run_scf()` | Build density-independent objects, then orchestrate the nonlinear SCF iteration. |
| `driver.py` | `run_single_point()` | Provide the short public `prepare + SCF` workflow. |
| `Output/parsec_output.py` | `ParsecTextReporter` | Format setup, iteration, eigenvalue, energy, convergence, and timing sections. |
| `cli.py` | `main()`, `save_result_archive()` | Resolve files, connect the reporter to the SCF callback, and serialize results. |

The root package re-exports most high-level component APIs, so callers normally
use `from new_architecture import ...`. Lower-level solver routines such as
`run_chebff()` and `run_subspace_filter()` are exported by
`new_architecture.Eigensolvers`; `save_result_archive()` lives in
`new_architecture.cli`.

## Single-point calculation flow

The modular path follows this sequence:

1. `parse_parsec_input()` translates the supported `parsec.in` labels into a
   `SinglePointInput` and resolves every required POTRE path. The CLI dry run
   and normal preparation then load and validate the POTRE contents.
2. `prepare_single_point()` optionally recenters the geometry, loads POTRE
   data, counts valence electrons, builds the active grid and sparse
   finite-difference operator, constructs local/nonlocal ionic terms, builds
   and normalizes the initial atomic density, and evaluates the ion-ion energy.
3. The initial density generates `V_H[ρ₀]` and
   `V_xc[ρ₀ + ρ_core]`, giving the first local field
   `V_in = V_ion,local + V_H + V_xc`.
4. `KohnShamHamiltonian` applies

   ```text
   H = -∇² + diag(V_ion,local + V_H + V_xc) + V_NL
   ```

   The Kleinman-Bylander nonlocal term is applied through sparse projector
   factors; a dense grid-by-grid nonlocal matrix is not stored.
5. On the first SCF iteration, `solve_eigval()` runs the explicitly requested
   CHEBFF or CHEBDAV implementation. Later iterations reuse the buffered Ritz
   space through exactly one translated SUBSPACE filter. An unported solver
   is rejected; no ARPACK or Fortran fallback is substituted.
6. `fermi_occupations()` fills the states and
   `density_from_orbitals()` constructs

   ```text
   ρ_i = (2 / h³) Σ_n f_n |q_i,n|²
   ```

   for Euclidean-normalized grid vectors `q`.
7. The new density produces a warm-started Hartree CG solve and a new CA-LDA
   evaluation. NLCC contributes to XC but not to the Hartree source.
8. With `R = V_out - V_in`, the default charge-weighted residual is

   ```text
   SRE = sqrt[(h³ / N_e) Σ_i ρ_i R_i²]
   ```

   `AndersonMixer` constructs the next input potential. SCF convergence is
   controlled by this potential residual (or the explicitly selected plain
   residual), not by energy, Hartree, or eigenpair residuals.
9. `total_energy()` uses the eigenpairs generated by `V_in` and the
   density-dependent output potentials:

   ```text
   E_total = E_band - ∫ρ(V_in - V_ion)dr
             + 1/2 ∫ρ V_H,out dr + E_xc,out + E_ion-ion
   ```

10. The result retains the input, raw output, and next mixed effective
    potentials together with densities, eigenpairs, occupations, energy
    components, and SCF history. The reporter writes the supported
    PARSEC-shaped text sections.

All physical arrays use PARSEC conventions:

- positions and radii: bohr;
- densities: electrons/bohr³;
- potentials, eigenvalues, and energies: Rydberg;
- wavefunction columns: Euclidean normalized on the discrete grid.

## Modular Python API

Parse a real PARSEC input, inspect all static terms, and then decide whether to
run SCF:

```python
from new_architecture import (
    parse_parsec_input,
    prepare_single_point,
    run_scf,
)

input_path = "src/new_architecture/benchmarks/h2_canonical_nodg/parsec.in"
translation = parse_parsec_input(input_path)
system = prepare_single_point(translation.problem)

print(system.grid.size)
print(system.grid.integrate(system.initial_density))
print(system.ionic_potential)
print(system.nonlocal_operator.labels)
print(system.ion_ion_energy)

hartree = system.solve_hartree(system.initial_density)
xc = system.evaluate_xc(system.initial_density)
hamiltonian = system.hamiltonian(
    system.ionic_potential + hartree.potential + xc.potential
)

result = run_scf(system)
print(result.converged, result.energies.total)
```

For the shortest complete program:

```python
from new_architecture import parse_parsec_input, run_single_point

input_path = "src/new_architecture/benchmarks/h2_canonical_nodg/parsec.in"
problem = parse_parsec_input(input_path).problem
result = run_single_point(problem)
```

Grid construction, finite differences, POTRE parsing, local/nonlocal ionic
terms, Hartree, CA-LDA, occupations, mixing, and energy evaluation can also be
called independently through the APIs in the table above.

## Supported physical scope

The current new architecture supports:

- isolated spherical and rectangular-box domains;
- the full active grid without point-group reduction;
- spin-unpolarized CA/PZ LDA;
- scalar norm-conserving Martins-new pseudopotentials through `l = 3`;
- optional nonlinear core correction;
- CHEBFF or CHEBDAV for the first eigensolve and SUBSPACE reuse afterward;
- multipole Hartree boundaries for spheres and exact direct-Coulomb boundary
  values for boxes;
- potential-mixed SCF with PARSEC-style SRE convergence.

It does not yet support periodic/slab/wire systems, MPI or symmetry wedges,
spin polarization, spin-orbit coupling, GGA/meta-GGA/hybrids, forces,
relaxation, molecular dynamics, DFT+U, external fields, k-points, complex
orbitals, PARSEC restart files, or Ono-Hirose `Double_Grid_Order > 1`.

The code favors source readability and component validation over production
performance. Large fine-grid molecules can use substantially more time and
memory than Fortran PARSEC because the Python path currently has no symmetry
reduction, MPI, or GPU backend.

## Benchmarks

The tracked cases under `src/new_architecture/benchmarks` include:

- `h2_canonical_nodg`: canonical PARSEC hydrogen POTRE data on the currently
  supported no-double-grid path.
- `h2_full_nonlocal`: matched full-pseudopotential H2 comparison with component
  and total-energy notes.
- `0d_benzene`: PARSEC benzene input, pseudopotentials, reference output, and
  comparison notes.
- `0d_naphthalene`: larger PARSEC benchmark input and pseudopotentials for
  continued validation.

Example commands from the repository root:

```bash
parsec-py src/new_architecture/benchmarks/h2_full_nonlocal/parsec.in --no-archive
parsec-py src/new_architecture/benchmarks/0d_benzene/parsec.in --no-archive
parsec-py src/new_architecture/benchmarks/0d_naphthalene/parsec.in --no-archive
```

See the README or `COMPARISON.md` inside each completed comparison directory
before interpreting small numerical differences.

## Testing the new architecture

The suite runs from a source checkout, since several tests assert on the
repository layout. From the repository root:

```bash
python -m unittest discover -s src/new_architecture/tests -p "test_*.py" -v
```

The suite covers input translation, architecture boundaries, grid and
finite-difference construction, pseudopotential parsing/splines, local and
nonlocal ionic terms, isolated Hartree boundaries and CG, CA-LDA,
occupations, mixing, energy bookkeeping, CHEBFF/CHEBDAV/SUBSPACE behavior, CLI
output, and complete small SCF paths.

## Old architecture

The older implementation remains usable and is intentionally isolated under
`src/old_architecture`. Its recommended refactored entry point is:

```bash
python src/old_architecture/main_new.py --cpu path/to/input.in
```

With no input path, it starts the manual input flow. Major modules are:

| Module or folder | Responsibility |
|---|---|
| `main_new.py` | Small launcher that wires input, setup, backend, output, and solver stages. |
| `main.py` | Preserved original monolithic script; treat it as an executable script, not a library API. |
| `main_version2.py` and `rsdft_solver_version2.py` | Alternate experimental legacy driver/solver pair; not the recommended default. |
| `rsdft_models.py` | Legacy settings, prepared-system, backend, energy, diagnostic, and result dataclasses. |
| `rsdft_input.py` | Manual/file input, element metadata, geometry/density-grid loading, and unit conversion. |
| `rsdft_setup.py` | Grid/radius selection, recentering, output naming, and prepared-system construction. |
| `rsdft_backend.py` | CPU/GPU implementation selection and optional native ionic-kernel selection. |
| `rsdft_solver.py` | Legacy initial-density, Hamiltonian, eigensolver, Hartree/XC, mixing, energy, and SCF pipeline. |
| `rsdft_hartree.py` | Selectable legacy Hartree right-hand side and Poisson solve helpers. |
| `rsdft_output.py` and `rsdft_diagnostics.py` | Logs, density/wavefunction output, timing, and consistency diagnostics. |
| `Eigensolvers/`, `Laplacian/`, `Mixer/`, `Splines/`, `V_ion/`, `V_xc/` | Original CPU/GPU numerical routines grouped by topic. |
| `native/` | Implemented pybind11/C++17/OpenMP accelerators for `pseudoDiag` and `pseudoNL`. |
| `GUI/` and `Tools/` | Legacy input generation, timing extraction, and plots. |

The core legacy path additionally uses pandas and Matplotlib:

```bash
python -m pip install -e ".[legacy]"
```

GPU execution requires a CuPy build compatible with the installed CUDA
runtime. The GUI also requires a Python installation with Tk/Tkinter support.

### Legacy native extension

The `rsdft_native` C++/OpenMP extension is a separate, optional distribution
under `src/old_architecture/native`, so the main install stays pure Python.
Building it requires CMake 3.24 or newer, Ninja, and a C++17 compiler with
OpenMP support (for example, the appropriate MSVC Build Tools on Windows):

```bash
python -m pip install -v ./src/old_architecture/native
```

Set the OpenMP thread count with `OMP_NUM_THREADS`. The environment variables
`PARSEC_NATIVE_PSEUDODIAG` and `PARSEC_NATIVE_PSEUDONL` control the two native
ionic kernels independently.

Important: the current legacy backend selects its lazily loaded native wrappers
when these variables are unset. Therefore, either build `rsdft_native` before
running `main_new.py`, or explicitly set both variables to `0` to use the
older pure-Python ionic routines:

```bash
export PARSEC_NATIVE_PSEUDODIAG=0    # Windows PowerShell: $env:PARSEC_NATIVE_PSEUDODIAG = "0"
export PARSEC_NATIVE_PSEUDONL=0      # Windows PowerShell: $env:PARSEC_NATIVE_PSEUDONL = "0"
python src/old_architecture/main_new.py --cpu path/to/input.in
```

`--cpu` selects the CPU numerical backend; it does not by itself disable the
C++/OpenMP ionic kernels.

See [`src/old_architecture/README.md`](src/old_architecture/README.md) for
legacy-specific details.

## UPF-to-PARSEC conversion tool

`parsec_tools.upf_to_parsec` converts a conservative subset of norm-conserving
semilocal UPF v2 pseudopotentials to PARSEC Martins-new format. The input must
use `pseudo_type="SL"`, provide one scalar semilocal potential and one
`PP_CHI` reference function for every consecutive angular channel, and use a
pure-exponential source mesh:

```bash
upf-to-parsec input.UPF X_POTRE.DAT
```

Without installing, run it as a script with
`python src/parsec_tools/upf_to_parsec.py input.UPF X_POTRE.DAT`.

Useful options include `--xc-code`, `--grid-refinement {1,2}`,
`--allow-ionized-reference`, and `--force`. The converter resamples onto
PARSEC's shifted-log radial mesh and validates the written file. It rejects
formats it cannot convert without guessing, including general Kleinman-Bylander
`pseudo_type="NC"` files, PAW, ultrasoft, spin-orbit, NLCC, and
non-pure-exponential UPF meshes. Pair both the reported `Local_Component` and
`Correlation_Type` values with the generated potential in `parsec.in`.

## Further documentation

- [`src/new_architecture/README.md`](src/new_architecture/README.md): complete
  CLI, component API, benchmark, default, and limitation details.
- [`src/new_architecture/ARCHITECTURE.md`](src/new_architecture/ARCHITECTURE.md):
  package boundaries and native-port rules.
- [`src/new_architecture/PARSEC_ALGORITHM.md`](src/new_architecture/PARSEC_ALGORITHM.md):
  reviewed Fortran call path, formulas, defaults, and source locations.
- [`src/new_architecture/PYTHON_IMPLEMENTATION.md`](src/new_architecture/PYTHON_IMPLEMENTATION.md):
  Python implementation map and detailed component examples.
- [`src/new_architecture/provenance/source_map.json`](src/new_architecture/provenance/source_map.json):
  machine-readable Fortran-to-Python implementation status.

## Citing PARSEC.py

Z. Zhang, C. M. Perez, P. Kwon, M. Head-Gordon, and J. Qian, “ PARSEC.py: A Python-Based Real-Space Kohn–Sham Density Functional Theory Code Accelerated by Machine Learned Charge Density,” Journal of Computational Chemistry 47, no. 23 (2026): e70482, https://doi.org/10.1002/jcc.70482.

@article{https://doi.org/10.1002/jcc.70482,
author = {Zhang, Zeyi and Perez, Carlos Mora and Kwon, Patrick and Head-Gordon, Martin and Qian, Jin},
title = {PARSEC.py: A Python-Based Real-Space Kohn–Sham Density Functional Theory Code Accelerated by Machine Learned Charge Density},
journal = {Journal of Computational Chemistry},
volume = {47},
number = {23},
pages = {e70482},
keywords = {electronic structure theory, high performance computing, machine learning charge density, real-space KS-DFT},
doi = {https://doi.org/10.1002/jcc.70482},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.70482},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/jcc.70482},
abstract = {ABSTRACT PARSEC.py is a Python-based real-space Kohn–Sham density functional theory (real-space KS-DFT) framework designed to provide a user- and developer-friendly platform for first-principles electronic-structure simulations. Discretization on real-space grids eliminates basis-set approximations, while enabling systematic control of numerical accuracy and large-scale parallelization. Building upon the theoretical foundations of the original PARSEC code, the framework leverages the Python scientific ecosystem to support modular development and integration with modern machine learning (ML) tools. PARSEC.py incorporates ML-predicted density as an alternative initial guess for the superposition of atomic density (SAD) in self-consistent field (SCF) iterations. This capability provides a framework for integrating ML models into electronic-structure calculations and offers a pathway for accelerating SCF convergence. We present the theoretical framework, key algorithms, and performance characteristics of PARSEC.py using both SAD and ML initializations.},
year = {2026}
}

## License

See [`LICENSE`](LICENSE).
