# Python Single-Point Implementation

## Design goal

`src/new_architecture` is now independent of the older monolithic Python
SCF stack. Each physical operation has a public API, and `prepare_single_point`
stops after static setup so those terms can be inspected or reused without
running an SCF calculation.

All public physics arrays follow PARSEC units:

- coordinates/radii: bohr
- densities: electrons/bohr³
- potentials/eigenvalues/energies: Rydberg
- orbitals: Euclidean-normalized columns

## Module structure

```text
new_architecture/
├── Input/parsec_input.py      ESDF parsing and validated settings
├── Grid/cluster.py            full active isolated domain and index map
├── Laplacian/finite_difference.py
│                              Fornberg weights and sparse -∇²
├── Pseudopotential/potre.py   Martins-new POTRE reader/radial data
├── V_ion/ionic_potential.py   local Vion, SAD/NLCC, KB projectors, EII
├── Hartree/poisson.py         boundary multipoles and Poisson CG
├── V_xc/ca_lda.py             unpolarized CA/PZ LDA
├── Hamiltonian/operator.py    kinetic/local/nonlocal matvec composition
├── Eigensolvers/
│   ├── spectral_bounds.py     non-BETA short Lanczos bound
│   ├── lapack_random.py       pure-Python DLARNV uniform stream
│   ├── chebyshev.py           normalized block recurrences
│   ├── orthogonalize.py       PARSEC orth_normal decisions
│   ├── rayleigh_ritz.py       projection and rotation
│   ├── chebff.py              first-SCF fixed filter
│   ├── chebdav.py             first-SCF lock/expand/restart solver
│   ├── subspace.py            one later-SCF saved-subspace filter
│   └── eigval.py              selected first solver/SUBSPACE dispatch
├── Occupations/fermi_dirac.py Fermi level and 2/h³ density
├── Mixer/anderson.py          SRE metrics and Anderson mixing
├── Energy/total_energy.py     PARSEC totnrg expression
├── SCF/single_point.py        static preparation and SCF loop
├── Output/parsec_output.py    PARSEC-style report formatter
├── driver.py                  short two-stage public workflow
├── models.py                  typed inputs, settings, and results
├── cli.py                    logging and result-archive command runner
├── main.py                   folder-local command launcher
├── provenance/source_map.json per-component source/status manifest
└── tests/                    unit, architecture, and integration tests
```

Numerical code lives only in the concept packages above. The former flat
compatibility modules were removed, leaving one readable implementation for
each operation.

## PARSEC-file command line

The `new_architecture/main.py` launcher is contained inside this package
and leaves the older `src/old_architecture/main.py` workflow separate. With
`parsec.in` and `H_POTRE.DAT` in the same directory:

```powershell
cd src\new_architecture
python -m pip install -r requirements.txt
python main.py path\to\parsec.in --dry-run
python main.py path\to\parsec.in
```

The parser translates supported isolated single-point ESDF labels into the
same `SinglePointInput` models used by the component API. It resolves
`<Atom_Type>_POTRE.DAT` next to the input, or from an explicit `--pp-dir`.
The dry run parses the POTRE contents as well as the input but does not build
the real-space grid.

The full command creates `parsec.out` and
`parsec_python_results.npz` beside the input. The archive contains atom
metadata, grid coordinates, densities, local potentials, eigenpairs,
occupations, all energy terms, and per-iteration convergence data.
Wavefunctions are included when `Output_All_States` is true or
`--save-wavefunctions` is supplied.

`parsec.out` mirrors PARSEC's supported report sections and exact
energy/SRE labels. It prints plain SRE first and charge-weighted SRE second,
matching PARSEC. Python-only limitations are stated in the report; symmetry
tables, forces, dipoles, and MPI statistics are not fabricated.

Useful controls:

```text
--pp-dir DIR          explicit pseudopotential location
--output FILE         archive path (a .npz suffix is enforced)
--log FILE            text log path
--no-archive          skip the NumPy archive
--save-wavefunctions  force wavefunction storage
--quiet               suppress console progress
--debug               show a traceback for runtime failures
```

The launcher refuses output paths that would overwrite `parsec.in`, a POTRE
file, or one another. It returns 0 for convergence, 3 when a valid calculation
finishes `Max_Iter` without convergence, 2 for input errors, 1 for other
runtime failures, and 130 when interrupted.

## Complete calculation

The following Windows example uses a POTRE file stored inside WSL. WSL must be
running for the UNC path to be available.

```python
import numpy as np

from new_architecture import (
    Atom,
    EigensolverSettings,
    GridSettings,
    MixingSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
    run_single_point,
)

bohr_per_angstrom = 1.8897268

problem = SinglePointInput(
    atoms=[
        Atom("H", np.array([0.0, 0.0, -0.375]) * bohr_per_angstrom),
        Atom("H", np.array([0.0, 0.0,  0.375]) * bohr_per_angstrom),
    ],
    pseudopotentials={
        "H": SpeciesPotential(
            path=(
                r"\\wsl.localhost\Ubuntu-24.04\home\zeyizhang"
                r"\PARSEC\tests\H2\python_pp\H_POTRE.DAT"
            ),
            local_angular_momentum=0,  # s
            read_valence_density=True,
        )
    },
    grid=GridSettings(
        spacing=0.20 * bohr_per_angstrom,
        radius=7.0 * bohr_per_angstrom,
        expansion_order=8,
        shift=(0.5, 0.5, 0.5),
    ),
    scf=SCFSettings(
        number_of_states=16,
        fermi_temperature_kelvin=500.0,
        max_iterations=50,
        convergence_criterion=2.0e-4,
    ),
    eigensolver=EigensolverSettings(
        method="chebff",
        tolerance=1.0e-4,
        first_filter_degree=10,
        first_filter_cycles=2,
        matvec_block_size=6,
        filter_degree=10,
        filter_degree_delta=0,
    ),
    mixing=MixingSettings(parameter=0.15),
)

result = run_single_point(
    problem,
    callback=lambda step: print(
        step.iteration,
        step.weighted_residual,
        step.energies.total,
    ),
)

print(result.converged)
print(result.energies.total, "Ry")
```

The example mirrors the physical grid, state, temperature, convergence,
filter-degree, and mixing settings in PARSEC's
`tests/H2/python_pp/parsec.in`. The first solve is native CHEBFF and later
solves reuse its saved working subspace. The local component is an explicit
per-species choice because POTRE does not store it.

The result distinguishes two potentials:

- `input_effective_potential` generated the returned eigenpairs.
- `output_effective_potential` is the raw
  `Vion + VH[rho_new] + Vxc[rho_new]` before mixing.
- `next_effective_potential` is Anderson's mixed guess for another iteration.

Returned `hartree_potential` and `xc_potential` are output-side quantities. This
distinction is important for an unconverged calculation; only at self
consistency do the two effective potentials agree.

## Preparing without running SCF

```python
from new_architecture import prepare_single_point

system = prepare_single_point(problem)

print(system.grid.size)
print(system.grid.integrate(system.initial_density))
print(system.ionic_potential)
print(system.nonlocal_operator.labels)
print(system.ion_ion_energy)
```

`PreparedSinglePointSystem` contains all static objects and provides:

```python
hartree = system.solve_hartree(system.initial_density)
xc = system.evaluate_xc(system.initial_density)
hamiltonian = system.hamiltonian(
    system.ionic_potential + hartree.potential + xc.potential
)
```

`run_scf(system)` accepts the prepared object, allowing callers to modify or
inspect static terms first.

## Independent component examples

### Grid and finite-difference operator

```python
from new_architecture import (
    GridSettings,
    build_cluster_grid,
    build_negative_laplacian,
    second_derivative_coefficients,
)

settings = GridSettings(
    spacing=0.35,
    radius=8.0,
    expansion_order=12,
)
grid = build_cluster_grid(settings)
minus_laplacian = build_negative_laplacian(grid)
weights = second_derivative_coefficients(12)
```

`minus_laplacian` is the physical PARSEC kinetic matrix in Rydberg. Missing
neighbors multiply zero. The same matrix is passed to Poisson; only the Poisson
right-hand side receives nonzero boundary corrections.

### POTRE radial data

```python
from new_architecture import ParsecRadialSpline, read_parsec_pseudopotential

pp = read_parsec_pseudopotential("O_POTRE.DAT")
v_local = pp.local_potential(distances, local_l=1)
v_local_spline = pp.local_potential(
    distances,
    local_l=1,
    use_spline=True,
    spline_padding_width=grid.settings.stencil_half_width,
)
beta_s, denominator_sign = pp.radial_projector(
    angular_momentum=0,
    local_l=1,
)

# The interpolation kernel is independently reusable and inspectable.
local_spline = ParsecRadialSpline.from_positive_grid(
    pp.radii,
    pp.channel_potentials[1],
    padding_width=grid.settings.stencil_half_width,
)
values_at_selected_radii = local_spline(distances)
```

The reader retains radial channels, wavefunctions, occupations, cutoffs,
valence density, and optional core density. It validates radial monotonicity and
declared channel count. `ParsecRadialSpline` is a direct NumPy translation of
PARSEC's clamped `spline.f90` recurrence, including the artificial
negative-radius padding tied to `Expansion_Order/2`.

### Hartree only

```python
from new_architecture import HartreeSettings, solve_hartree

solution = solve_hartree(
    density,
    grid,
    minus_laplacian,
    HartreeSettings(multipole_order=9),
    initial_potential=guess,
    raise_on_nonconvergence=False,
)

print(solution.converged, solution.residual_norm)
```

The standalone function raises on Poisson nonconvergence by default so an SCF
cannot silently continue with a failed potential. Pass
`raise_on_nonconvergence=False`, as above, when diagnosing or inspecting a
partial CG result. PARSEC itself only prints a warning.

`HartreeSettings.boundary_method="auto"` uses multipoles for a sphere and exact
discrete Coulomb boundary values for a box. The latter corresponds to
PARSEC's `Full_Hartree` sum and is much more expensive. An origin-centered
multipole series is deliberately rejected at general box faces because charge
near a corner can lie outside its convergence radius.

### CA-LDA only

```python
from new_architecture import ca_lda

xc = ca_lda(
    valence_density,
    grid.volume_element,
    core_density=core_density,
)

print(xc.potential)
print(xc.total_energy)
```

### Local/nonlocal Hamiltonian pieces

`KohnShamHamiltonian` exposes `apply_kinetic`, `apply_local`,
`apply_nonlocal`, and `apply`. `NonlocalProjectorOperator` exposes the sparse
projector matrix, labels `(atom,l,m)`, its low-rank action, and an explicit
sparse matrix for small validation problems.

## Input and default behavior

The settings dataclasses use the relevant PARSEC defaults where they are
well-defined:

| Python setting | Default |
|---|---:|
| `GridSettings.expansion_order` | 12 |
| `GridSettings.shift` | `(0.5,0.5,0.5)` |
| Hartree multipole order | 9 |
| Hartree boundary method | `auto` |
| Hartree relative/absolute tolerance | `1e-7` / `1e-13` |
| Hartree max matrix-vector products | 1600 |
| SCF max iterations | 50 |
| weighted SRE convergence | `2e-4 Ry` |
| Fermi temperature | 80 K |
| Anderson parameter/memory/restart | 0.30 / 4 / 20 |
| diagonalization tolerance | `1e-4` |
| later Chebyshev degree/delta | 15 / 3 |
| state buffer | 6 |

If `SCFSettings.number_of_states` is omitted, Python chooses
`ceil(Ne/2)+6`. PARSEC normally expects `States_Num` from the input, so explicit
state counts are preferable for reproducible comparisons.

`SpeciesPotential.read_valence_density` defaults to false, matching PARSEC's
`Read_VCD` default. In that mode the atomic starting density is reconstructed
from POTRE occupations and pseudo-wavefunctions. Set it true to use the
tabulated valence charge, as the H2 reference input does.

`recenter_geometry=True` applies PARSEC's bounding-box-midpoint recentering.
Set it false when coordinates are already deliberately placed relative to the
domain origin.

## Exact correspondences

The following behaviors were translated directly:

- half-shifted uniform cluster lattice and inclusive shape mask
- centered Fornberg coefficients for orders 2 through 20
- zero exterior wavefunction samples
- Rydberg `-∇²` kinetic convention
- Martins-new POTRE layout and `rV` conversion
- local-channel selection and `-2Z/r` tail
- file/reconstructed atomic starting density and global normalization
- NLCC used for XC but not Hartree
- normalized KB projectors with explicit denominator sign and `h³`
- multipole Hartree boundary folded into all missing stencil neighbors
- exact direct Coulomb boundary for non-spherical box faces
- unpreconditioned Poisson CG stopping rule
- CA/PZ LDA formulas and constants
- spin-degenerate Fermi occupations and `2/h³` density
- input-Hxc/new-density total-energy correction
- weighted/plain potential SRE
- Anderson potential mixing
- Chebyshev normalized recurrence and degree split

## Deliberate scope and differences

### Full grid instead of symmetry wedge

Python keeps every active point and applies identity symmetry. This is
physically the same unreduced Hamiltonian but can be much larger than PARSEC's
default wedge. For the H2O reference:

- half-shifted full grid: 179,944 points
- PARSEC symmetry wedge: 44,986 points

The Python `shift` is independent of symmetry. This avoids copying PARSEC's
`Ignore_Symmetry -> zero shift` coupling.

### Eigensolver initialization

For `Eigensolver: chebff`, the SCF path is now a native Python translation of
the audited non-`BETA` policy:

1. add `Subspace_Buffer_Size` working states;
2. generate the first basis with the LAPACK `DLARNV(IDIST=2)` stream used by
   PARSEC;
3. obtain the initial bound with the non-reorthogonalized short Lanczos
   recurrence;
4. run the fixed number of CHEBFF filter/orthogonalize/Rayleigh--Ritz cycles;
5. save all working Ritz pairs; and
6. on every later SCF iteration, perform exactly one `subspace.f90z` filter
   and Rayleigh--Ritz rotation.

There is no Ritz-residual acceptance test and no ARPACK fallback in this path.
Residuals calculated after later rotations are diagnostics only. The
source-level cross-block `sigma` carry is the default, so the configured
`Matvec_Blocksize` remains part of the numerical trajectory.

For `Eigensolver: chebdav`, the first solve instead uses the native real,
non-`BETA` Chebyshev--Davidson path. It incrementally extends the projected
Hamiltonian, locks only a contiguous low-energy Ritz prefix satisfying
`Diag_Tolerance * max(abs(Ritz))`, performs PARSEC's inner/outer restarts, and
runs the same approximate filtered-subspace cleanup if restart limits are
reached. Its 12-column Davidson work window is distinct from the six safety
states returned to later filtering. Subsequent SCF iterations use the same
one-pass `subspace.f90z` translation as CHEBFF.

`arpack` remains an explicit but unported selection and is rejected rather
than substituted.

### Currently unsupported

- periodic, slab, and wire boundary conditions
- point-group reduction and MPI distribution
- ellipsoid and cylinder masks in the Python builder
- spin polarization and spin-orbit channels
- GGA/meta-GGA/hybrid functionals
- Ono-Hirose `Double_grid_order > 1`
- forces, stress, geometry optimization, and molecular dynamics
- DFT+U, external fields, k-points, and complex orbitals
- old Martins, Wang, and FHI pseudopotential formats
- PARSEC binary restart/checkpoint compatibility

Sphere and box domains are implemented. POTRE files with spin-orbit channels
or a non-CA/PZ XC label are rejected rather than silently misused.

## Performance expectations

The implementation prioritizes readable, separately testable algorithms.
Sparse SciPy matrices and sparse low-rank projector storage are used, but there
is no MPI, symmetry reduction, or GPU path. Fine molecular grids can therefore
be substantially slower and more memory-intensive than production PARSEC.

Sphere Hartree boundaries use the inexpensive multipole expansion. Box
boundaries use the correct direct Coulomb sum in `auto` mode; its
`O(N_grid*N_boundary)` cost is suitable mainly for small reference systems.

Component inspection and small/medium reference calculations are the intended
first use. A later performance layer can replace matrix construction,
projector application, Poisson CG, or eigensolver internals behind the existing
interfaces.

## Validation performed

The focused test suite checks:

- exact H2 `parsec.in` translation and adjacent POTRE discovery
- folder-local launcher and package CLI dry runs
- a complete one-iteration CLI calculation through result handoff
- result-archive suffix and atom metadata
- H2O reference active-grid counts for half and zero shifts
- order-eight Fornberg values and sparse-matrix symmetry
- CA potential against the derivative of `rho*epsilon_xc`
- finite-temperature electron count and discrete density normalization
- equal 0 K occupation across a degenerate frontier
- Rydberg factor two in the monopole Hartree boundary
- all moments through `l=9` against a direct far-field sum
- exact direct box boundary and Poisson CG stopping residual
- weighted/plain residual construction
- first and history-dependent Anderson mixing steps
- POTRE sections, tail, and KB projector normalization path
- fixed-cycle CHEBFF followed by exactly one later subspace filter
- CHEBDAV residual-prefix locking, restart/cleanup, and later SUBSPACE reuse
- source-compatible block-degree split and cross-block recurrence state
- explicit rejection of unported ARPACK with no substitution
- architecture checks forbidding subprocess, WSL, ctypes, cffi, and f2py
- total-energy input-Hxc double-counting arithmetic
- nonlinear-core density entering XC while remaining separate from valence
- one complete setup/eigensolver/Hartree/SCF iteration

The actual WSL O POTRE fixture was also read directly:

- symbol/XC: `O / ca`
- 2 channels and 1,027 positive-radius samples
- ionic valence: 6
- integrated stored valence density: approximately `6.000164`

An actual H2 setup using PARSEC's H POTRE file also produced:

- 179,944 full active grid points
- 4,267,480 sparse Laplacian entries
- initial-density integral `1.9999999999999984` electrons
- 6 nonlocal projector columns for two H atoms
- ion-ion energy `1.41113872 Ry`; PARSEC prints `1.41113867 Ry`

Run validation with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s src\new_architecture\tests -v
```

## Suggested next comparison sequence

For a strict Python-versus-PARSEC campaign, compare in this order:

1. identical atom positions after recentering
2. identical grid shift, integer coordinates, and active count
3. finite-difference coefficient arrays
4. local ionic potential at selected grid rows
5. each `(atom,l,m)` projector and its KB action on a fixed vector
6. normalized initial atomic density
7. initial multipoles, Poisson RHS, and Hartree potential
8. pointwise CA-LDA potential and total `E_xc`
9. Hamiltonian action on fixed vectors
10. first eigenvalues/eigen-residuals
11. occupations and new density
12. raw output potential, SRE, mixed potential, and energy terms per iteration

This ordering localizes the first disagreement instead of treating total energy
as the only diagnostic.
