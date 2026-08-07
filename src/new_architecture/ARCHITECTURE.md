# Native Python PARSEC port architecture

## Purpose

This tree is the readable, concept-oriented home for a native Python
translation of the PARSEC isolated single-point calculation. "Native Python"
means the calculation does not invoke or bind to the PARSEC Fortran program.
PARSEC source is the algorithm specification and reference implementation, not
a runtime dependency.

Each numerical operation has one implementation in the concept packages
below. The redundant flat compatibility modules have been removed. A concept
package being present still does not imply literal PARSEC parity; current
claims are recorded in
[`provenance/source_map.json`](provenance/source_map.json).

## Package map

```text
main.py
  -> cli.py
       -> Input
       -> Output
       -> driver.py
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
```

The flow is a dependency direction, not permission for lower-level modules to
import the SCF driver. Physics components must remain independently callable.

## Responsibilities

| Package | Owns | Does not own |
|---|---|---|
| `Input` | ESDF parsing and validated calculation configuration | grid construction or file output |
| `Grid` | isolated-domain points, masks, shifts, and index maps | kinetic coefficients |
| `Laplacian` | finite-difference coefficients and kinetic application | effective potentials |
| `Pseudopotential` | POTRE parsing, radial data, interpolation-ready models | placement on a molecular grid |
| `V_ion` | local potential, KB projectors, atomic density, ion-ion energy | SCF state |
| `Hartree` | isolated boundary values and Poisson solution | density mixing |
| `V_xc` | CA-LDA potential and energy density | total-energy assembly |
| `Hamiltonian` | composition and block matrix-vector products | choosing an eigensolver |
| `Eigensolvers` | PARSEC-selected eigensolver control flow and reusable subspaces | occupations or SCF mixing |
| `Occupations` | Fermi level, occupations, and density from orbitals | eigensolution |
| `Mixer` | explicit mixer history and PARSEC SRE metrics | convergence policy selection |
| `Energy` | named Rydberg energy components and total energy | text formatting |
| `SCF` | ordering the component calls and applying stopping policy | hidden algorithm substitution |
| `Output` | PARSEC-shaped reporting and optional serialization | numerical decisions |

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
