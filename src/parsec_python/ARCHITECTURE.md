# Native Python PARSEC port architecture

## Purpose

This tree is the canonical home for a native Python translation of the PARSEC
isolated single-point calculation, of its forces, and of the ionic steps
(structural relaxation and molecular dynamics) built on them. Readable, concept-oriented physics modules
and their optimized execution backends coexist in one package. "Native
Python" means the calculation does not invoke or bind to the PARSEC Fortran
program; optional C++/OpenMP and CUDA kernels implement Python-owned
algorithms. PARSEC source is the specification and reference, not a runtime
dependency.

Each physical operation has one readable definition in the concept packages
below. Performance counterparts live only under `acceleration/` and must pass
parity tests against that definition. The redundant flat compatibility
modules have been removed. Current parity claims are recorded in
[`provenance/source_map.json`](provenance/source_map.json).

## Package map

```text
main.py / python -m parsec_python
  -> acceleration/cli.py
       -> Input
       -> Output
       -> acceleration/driver.py
            -> SCF
                 -> Grid
                 -> Laplacian
                 -> Pseudopotential
                 -> V_ion
                 -> Hartree
                 -> V_xc
                 -> Hamiltonian
                 -> Eigensolvers
                 -> Occupations
                 -> Mixer
                 -> Energy
            -> Forces
            -> Relaxation / MolecularDynamics (ionic loop around SCF and Forces)
            -> acceleration/backends
            -> acceleration/Symmetry
            -> acceleration/Eigensolvers

reference_main.py
  -> cli.py -> driver.py -> SCF
```

K-point and band-path parallelism follows the same rule. The readable
definition is serial: `SCF/kpoint_solver.py` (`SerialKPointSolver`, `map_tasks`)
is all the k-point, spin-orbit, non-collinear and band-path drivers import.
`acceleration/SCF/kpoint_parallel.py` and `acceleration/Parallel/` (worker
threads, BLAS control, GPU device workers, `acceleration/Hamiltonian/device_bloch.py`)
are the performance counterpart; importing `parsec_python.acceleration` installs
it with `set_parallel_backend`. `acceleration/tests/test_kpoint_parallel.py`
checks it against the serial definition.

The flow is a dependency direction, not permission for lower-level modules to
import the SCF driver. Physics components must remain independently callable.

## Responsibilities

| Package | Owns | Does not own |
|---|---|---|
| `Input` | ESDF parsing and validated calculation configuration | grid construction or file output |
| `Grid` | isolated-domain points, masks, shifts, and index maps | kinetic coefficients |
| `Laplacian` | finite-difference coefficients and kinetic application | effective potentials |
| `Pseudopotential` | POTRE and old-format POTR parsing, radial data, interpolation-ready models | placement on a molecular grid |
| `V_ion` | local potential, KB projectors, atomic density, ion-ion energy | SCF state |
| `Hartree` | isolated boundary values and Poisson solution | density mixing |
| `V_xc` | CA/PZ-LDA and PBE potentials and energy densities, unpolarized and spin-polarized | total-energy assembly |
| `Hamiltonian` | composition and block matrix-vector products | choosing an eigensolver |
| `Eigensolvers` | PARSEC-selected eigensolver control flow and reusable subspaces | occupations or SCF mixing |
| `Occupations` | Fermi level, occupations (searched, or fixed by `occup.in`), and density from orbitals | eigensolution |
| `Mixer` | explicit mixer history and PARSEC SRE metrics | convergence policy selection |
| `Energy` | named Rydberg energy components and total energy | text formatting |
| `SCF` | ordering the component calls and applying stopping policy | hidden algorithm substitution |
| `Forces` | ion-ion, local, nonlocal forces, their symmetrization and net-force removal | moving atoms |
| `Relaxation` | L-BFGS-B, steepest-descent and manual ionic steps, the ionic loop, relaxation files | electronic structure of a structure |
| `MolecularDynamics` | the integrator of `moldyn.f90`, its input, random numbers and files | energies or forces |
| `Output` | PARSEC-shaped reporting and optional serialization | numerical decisions |
| `acceleration` | backend selection, symmetry sectors, GPU eigensolvers, native kernels, and timing provenance | independent physical formulas or silent accuracy changes |

## Public API rules

1. A numerical module exposes a small, typed API with documented array shapes,
   normalization, units, boundary assumptions, and mutations.
2. Lower-level routines return values or result dataclasses. They do not print,
   open output files, or mutate module-global state.
3. The SCF driver composes components; it does not duplicate their numerical
   formulas.
4. Reporting observes structured results and never changes a calculation.
5. A requested solver is the solver that runs. There is no silent ARPACK,
   dense-diagonalization, or other recovery path.
6. Random initial subspaces accept an explicit seed or generator.
7. Algorithm variants are named by behavior, not `_new`, `_original`, or
   `_version2`.
8. Each package has an `__init__.py`, but it exports only APIs whose port status
   and tests support the claim.

## Source traceability

Every implemented algorithm should identify:

- the PARSEC file and routine it translates;
- relevant input defaults and conditional branches;
- the Python representation of each important Fortran array;
- intentional deviations, if any; and
- focused verification evidence.

Promotion from `implemented_unverified` to `verified_component` requires more
than an end-to-end energy comparison. Tests should cover intermediate arrays,
normalization, iteration counts or stopping decisions where deterministic,
and failure behavior.

## Implementation rule

New implementations enter their concept package directly and are tested
through component APIs. Redundant flat modules must not be introduced.
The root `__init__.py` may expose selected public objects, but their
implementations remain in the owning concept package. The driver composes
those APIs; numerical parity status is promoted only after source-level and
intermediate-array evidence is available.
