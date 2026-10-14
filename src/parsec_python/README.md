# `parsec_python` package

`parsec_python` is the canonical implementation of PARSEC-style isolated
real-space DFT in this repository. The package contains both readable
scientific components and accuracy-audited accelerated backends; users do not
need to choose between separate source trees.

The implementation is native Python/C++/CUDA. It does not launch the PARSEC
executable or call Fortran. PARSEC source and output are used as the algorithm
specification and numerical reference.

## Entry points

For Linux environment creation, C++/OpenMP builds, optional NVIDIA/CuPy setup
and Bash launch commands, see the repository's
[Linux installation guide](../../README.md#linux-installation).

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

All three commands select the optimized workflow. By default a calculation
runs on the GPU where one is usable and on the CPU otherwise; `--device gpu`
or `--device cpu` names the device, `--gpus N` (or `--gpu-ids`) the GPUs and
`--threads N` the CPU threads, and `parsec.out` reports them under
"Execution resources" (see "CPU or GPU, and how much of each" in the
repository README). `--backend auto` combines
the fastest compatible SciPy, native C++/OpenMP, and CuPy components.
`--symmetry auto` detects exact supported operations and applies
representation decomposition when profitable. Missing optional acceleration
causes a reported safe fallback.

Default output files beside `parsec.in` are:

- `parsec.out`, containing PARSEC-shaped setup, SCF, energy, convergence, and
  timing sections;
- `parsec_python_results.npz`, containing structured arrays and metadata,
  unless `--no-archive` is supplied.

Control the verbosity of the `Acceleration backend:` setup section in `parsec.in`:

```text
Output_Level: 1
```

The default level `1` prints a concise backend summary, including the device,
hybrid component placement, thread count when available, precision and fallback
warnings. Use `Output_Level: 2` (or higher) for all backend settings and diagnostic
details. This controls only that setup section: output from `SCF iter # 1` onward,
including per-iteration results, final energies and timing statistics, is unchanged.
It does not enable extra profiling or change calculation settings.

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
| `Pseudopotential/` | Read Martins-new `POTRE.DAT` and old-format `POTR.DAT`, integrate radial data, and reproduce PARSEC splines. |
| `V_ion/` | Assemble local ionic fields, KB nonlocal factors, initial atomic density, and ion-ion energy. |
| `Hartree/` | Construct open-boundary data and solve the finite-difference Poisson equation. |
| `V_xc/` | Evaluate CA/PZ LDA or PBE, without or with spin polarization, and their energy terms. |
| `Hamiltonian/` | Apply the matrix-free Kohn--Sham operator. |
| `Eigensolvers/` | Readable CHEBFF, CHEBDAV, subspace, filtering, orthogonalization, and Ritz primitives. |
| `Occupations/` | Determine Fermi occupations and rebuild the density. |
| `Mixer/` | Apply PARSEC-style potential mixing and residual tests. |
| `Energy/` | Assemble the total-energy decomposition. |
| `SCF/` | Orchestrate one complete self-consistent single point. |
| `Forces/` | Evaluate ion-ion, local and nonlocal atomic forces and symmetrize them. |
| `Relaxation/` | Move the atoms: L-BFGS-B, steepest descent, manual structures, the ionic loop and its files. |
| `MolecularDynamics/` | Integrate the equations of motion of `moldyn.f90` and write its files. |
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
- scalar norm-conserving Martins-new pseudopotentials through `l=3`, and
  old-format Martins files (`Pseudopotential_Format: martins_old`);
- occupations read from `occup.in` (`Fermi_Temp` negative) in place of a
  Fermi distribution;
- separable Kleinman--Bylander nonlocal projectors;
- optional nonlinear core correction;
- CA/PZ LDA and PBE;
- collinear spin polarization of isolated systems (`Spin_Polarization: true`)
  with both functionals, see [Spin polarization](#spin-polarization);
- one post-SCF evaluation of the B3LYP energy on the converged orbitals of an
  isolated system (`Energy_Refinement: b3lyp`), see
  [Energy refinement by a hybrid functional](#energy-refinement-by-a-hybrid-functional);
- potential-mixed SCF with PARSEC-style occupations and eigensolver policy;
- exact automatic reflection/signed-permutation representation reduction
  where supported;
- core-hole species labels such as `C-1s` with an explicit
  `Element_Symbol: C` and optional `Atomic_Energy_Correction`;
- atomic forces, structural relaxation (`Minimization: bfgs`, `simple`,
  `manual`) and molecular dynamics of isolated systems, on every execution
  path including several GPU nodes, with `relax_restart.dat` exchanged with
  PARSEC in both directions; see
  [FORCES_AND_RELAXATION.md](FORCES_AND_RELAXATION.md).

Not currently supported as production physics:

- periodic boundary conditions;
- spin polarization of periodic cells, spin--orbit coupling and
  non-collinear magnetism;
- forces, relaxation and molecular dynamics of periodic cells;
- self-consistent hybrid functionals, meta-GGA functionals and DFT+U;
- PARSEC restart files other than `relax_restart.dat` and `mdinit.dat`;
- Ono--Hirose double-grid order greater than one.

Unsupported requested input is rejected rather than silently approximated.
Sphere Hartree boundaries use a multipole expansion; box boundaries use the
exact, slower direct Coulomb construction.

The multipole boundary values differ from PARSEC's by default where PARSEC's
are inaccurate. The expansion of order `Solver_Lpole` converges like
`(r_atom/R)^l`, so on the sphere of a large cluster it omits a potential of
0.5 to 1 Ry, which binds a spurious empty state at the boundary and makes the
energy depend on the vacuum. A prepared calculation therefore estimates,
from the valence charges placed at the nuclei, the largest potential the
expansion omits on the sphere at every order. Where the estimate at
`Solver_Lpole` exceeds a tenth of `Hartree_Boundary_Tolerance` (default
`1e-3 Ry`), two things change:

- the order becomes the smallest one from `Solver_Lpole` up (at most 60)
  whose estimate is within the tolerance, so `Solver_Lpole` is the minimum
  order, not a fixed one;
- the terms of those point charges above the order in use, the *atomic
  tail*, are added to the boundary values once at set-up
  (`Hartree_Atomic_Tail: auto | true | false`, default `auto`).

Small molecules stay below the threshold and keep PARSEC's boundary bit for
bit (benzene, naphthalene, CH4, CF4, H2 and the nanodiamonds up to 17 atoms
among the shipped inputs). From about 70 atoms at 5 A of vacuum the tail is
applied, and from about 300 atoms the order rises as well (12 for C185H124,
16 for C459H204, 34 for C5659H1132, 36 for C9449H1572); `0d_Si28H36` runs at
order 10 with the tail. `Hartree_Boundary_Tolerance: off` restores PARSEC's
boundary for every system, and a comparison with Fortran PARSEC of such a
system needs that line. `parsec.out` states under "Real-space setup" the
order in use, the estimate at `Solver_Lpole` and at that order, and whether
the tail was applied. Its header echoes the three input values. The
environment switches `PARSEC_HARTREE_BOUNDARY`, `_BOUNDARY_TOLERANCE`,
`_ATOMIC_TAIL` and `PARSEC_HARTREE_LPOLE` belong to the accelerated driver,
whose set-up block then names each input value they replaced; the reference
command line ignores them and says so in a warning. To compare the two
drivers with a boundary other than the default, set the input keywords.

The tolerance does not control the energy where the vacuum is thin. The
estimate and the tail are about the atoms as point charges at the nuclei.
Electron density that reaches the sphere is covered by neither: at the
exterior points next to it its potential is no multipole series about the
origin, so no order removes what it leaves. With 5 A of vacuum the boundary
values of converged densities are within 7e-5 Ry of the direct Coulomb sum
and their share of the total energy is below 1.5e-8 Ry per valence electron;
with 3 A they are within 1.6e-4 Ry and the share is 0.8e-7 to 2.6e-7 Ry per
valence electron (C35H36 to C459H204, first-order analysis of converged
densities). Below about 2 A the default boundary is not closer to the exact
one than PARSEC's: for C5H12 with 2.0 A of vacuum the total energy is 2.5e-5
Ry above that of the exact boundary with the default (order 9 and the tail)
and 2.6e-6 Ry below it with PARSEC's, where the truncation error of the atoms
happened to cancel the error of the density at the sphere; with 1.5 A both
are 3e-5 Ry above it. `Full_Hartree: true` sums the boundary values exactly
and is the choice for a sphere that small; an energy at thin vacuum is
evidence for or against a boundary only when compared with it.

### Sphere radius left to the code

A sphere input may leave `Boundary_Sphere_Radius` out, or give `auto`. The
radius and the tolerance of the Hartree boundary then come from the system,
by the default rule (version 1, `Hartree/domain.py`). An input that gives
its radius is read as before: same radius, same boundary plan, same bits; it
only gains lines in `parsec.out`.

The one accuracy knob is `Domain_Energy_Tolerance` (default `1e-3 Ry`): the
estimated energy that the finite sphere and its boundary values add to the
total energy, half each. It is not the SCF criterion, which stays
`Convergence_Criterion` (default `2e-4 Ry`).

- **Wall.** Before the SCF the energy the wall adds is estimated from the
  geometry and each species' own free atom, the valence density of its
  pseudopotential file:
  `W(R) = sum_a integral outside the sphere of 4 |grad sqrt(rho_atom)|^2`.
  The radius is the smallest multiple of 0.1 ang with at least 3 ang of
  vacuum beyond the outermost atom and `W(R)` within half the tolerance. No
  constant of the rule belongs to a species.
- **Boundary values.** The order of the multipole expansion follows from
  the plan above, with the tolerance tightened with the electron count:
  `min(1e-3, (B/2) / (K sqrt(N_e)))` Ry, cut to two digits, with
  `K = 6.25e-3` from 4 ang of vacuum and `1.25e-2` below. Where order 60
  misses it the radius grows. A `Hartree_Boundary_Tolerance` line of the
  input wins; `Hartree_Boundary_Tolerance: auto` asks for the rule's
  tolerance beside a radius of the input as well.

`W` is an estimate and not a bound. For hydrogen-terminated carbon it is 5
to 13 times the measured wall energy, which costs those clusters about
0.7 ang of radius; for ten systems without hydrogen at the surface the
measured energy at the rule's radius was 0.01 (NaCl) to 0.71 (Mg2) of the
wall share, and for twenty more neutral molecules up to 0.95 (Be2). It is
too small where the highest level of the system lies above that of its
atoms: Be2 exceeds the share at tolerances of 5e-4 Ry and below, and
clusters of closed-shell metal atoms are expected to do worse. Three cases
are handled apart:

- a species whose file holds no complete density (its integral differs from
  the file's occupations by more than 2 %, for the table and for the
  pseudo-wavefunctions): its atoms get 7.5 ang of vacuum, a guess, a NOTE
  names the species, and the wall estimate of the others ends with
  "without H, which has no complete free-atom density";
- electrons no free atom holds (an anion, or a file generated for an ion in
  a neutral system): the vacuum `v` becomes `max(1.6 v, v + 2.5 ang)`, with
  a WARNING and no claim that the tolerance is met. With the highest
  occupied level above zero, as for OH- and H- in PBE, the energy has no
  limit for a large sphere;
- `Hartree_Boundary_Tolerance: off` or `Hartree_Atomic_Tail: off` without a
  radius is refused unless `Full_Hartree` is set: the radius is chosen for
  the controlled boundary (C795H300 with PARSEC's boundary at 5 ang of
  vacuum is 3.2e-2 Ry off).

The rule covers the total energy and the occupied levels. Empty levels need
more vacuum: the lowest empty level of C795H300 at 4 ang lies 2.9e-3 Ry
above its value at 8 ang.

`parsec.out` says what was chosen and why, under the radius of the grid
data, and ends the block with the input lines that give the same domain
again, bit for bit:

```text
 --- Radius is  31.747412 bohrs
 --- Radius 16.8 ang chosen by the default rule 1 (Domain_Energy_Tolerance = 1.0E-03 Ry)
 --- Outermost atom: H at 12.295 ang from the centre; vacuum beyond it 4.505 ang
 --- Energy the sphere adds, estimated from the free atoms: 4.4E-04 Ry (aim 5.0E-04 Ry; an estimate, not a bound)
 --- Free-atom densities: C valence table 4 e, H valence table 1 e; share of the estimate: C 23 %, H 77 %
 --- Radius set by: atomic densities
 --- The rule covers the total energy and the occupied levels; empty levels need more vacuum.
 --- These two lines reproduce this domain bit for bit (with the same PARSEC_HARTREE_* switches):
       Boundary_Sphere_Radius: 16.8 ang
       Hartree_Boundary_Tolerance: 1.0e-03 Ry
```

Copy both lines. An input with the radius alone has the fixed tolerance of
`1e-3 Ry`, which is looser from 6,400 electrons up: order 40 instead of 44
for C5659H1132 and 43 instead of 50 for C9449H1572. Along a series of
geometries, fix the printed radius: the rule moves it in steps of 0.1 ang,
each worth 1e-5 to 3e-5 Ry.

Under "Real-space setup" every sphere run, also one with a radius of its
own, states what its boundary values are estimated to leave, read from the
plan in use:

```text
 --- Energy left by the boundary values at order 19 (tolerance 1.0E-03 Ry, atomic tail): about 8.0E-05 Ry, calibrated bound 2.5E-04 Ry (aim 5.0E-04 Ry)
       calibration: hydrogen-terminated clusters, 176 to 23,768 electrons
```

"About" is `2.0e-3 sqrt(N_e) e(L)` and the bound `K sqrt(N_e) e(L)`, with
`e(L)` the estimate of the plan at its order; for PARSEC's values, which a
plan keeps where its estimate is within a tenth of the tolerance, the bound
is `5e-2 sqrt(N_e) e(Solver_Lpole)`. Both constants were measured on
hydrogen-terminated clusters at a grid spacing of 0.2 ang (0.20 to 0.40 ang
on 176 and 424 electrons) and on no other surface. A plan without tolerance
or without the tail, one at order 60, and one that keeps PARSEC's order and
has the tail all the same (`Hartree_Atomic_Tail: on` where the estimate is
within a tenth of the tolerance) gets no number: a WARNING where the rule
chose the radius, a NOTE otherwise. An input with a radius is also told `W`
of its sphere, with a NOTE where an estimate exceeds its share. For `W`
that NOTE is no verdict: a radius taken from the block after an earlier SCF
gets it on hydrogen-terminated carbon, where the free atoms give several
times the wall energy (2.9e-3 Ry for C795H300 at 16.1 ang, against 2.7e-4 Ry
interpolated from its runs at 3.5 and 4 ang of vacuum), and the estimate
after the SCF decides.

After a converged SCF the wall is estimated again, from the density itself:

```text
 Density at the sphere: the sphere adds an estimated 1.7E-04 Ry to the total energy (decay constant 0.82 /bohr: a factor 10 per 0.74 ang)
   Wall share of Domain_Energy_Tolerance (5.0E-04 Ry): within it.
   For this system the wall share would be met by:
       Boundary_Sphere_Radius: 16.1 ang
       Hartree_Boundary_Tolerance: 6.7e-04 Ry
```

The charge of shells 0.4 bohr thick, from 0.8 to 4.0 bohr inside the sphere,
is fitted to `G sinh^2(kappa s) / kappa^2`; the estimate is `G / (2 kappa)`.
On 43 runs of eleven systems with 3 ang of vacuum or more the measured
energy above the widest sphere was 0.31 to 0.93 of it (median 0.83), and
0.52 to 0.86 on grids of 0.15 to 0.5 ang; for O2 and NO the estimate was 6
and 16 % too low.
It is marked rough where the fit is poor or the decay constant sits at an
end of its scan, and is still printed. A NOTE follows where it exceeds the
wall share and a WARNING where it exceeds the whole tolerance. The radius
underneath is the one the wall asks for, at most 2.5 ang from the radius of
the run ("at least" where that limit acts) and never below the minimum
vacuum or the stencil of the rule: a second calculation with it recovers
what the estimate before the SCF gave away, 10 to 12 % of the grid points
for C795H300. A run that did not converge gets no estimate.

Where the density did not set that radius, a comment behind it says what
did:

```text
       Boundary_Sphere_Radius: 34.2 ang   # order 60 misses the tolerance below this radius; the wall alone would allow 33.8 ang
       Hartree_Boundary_Tolerance: 2.0e-04 Ry
```

(the atoms of C9449H1572 at the radius of the rule with a made-up density
of 3e-5 Ry of wall energy; the calculation itself, on 16 A100, printed 34.2
ang and 33.9 ang for the wall alone). The largest order is asked for the tolerance of the rule only,
the one printed underneath, and can take the radius further than 2.5 ang
from that of the run. With a `Hartree_Boundary_Tolerance` line in the input
the radius is the wall's alone, the tolerance is not repeated, and what the
orders make of it is said by the set-up of a run at that radius. The
comment also names the minimum vacuum of 3 ang, the stencil, and a step off
the grid points.

The search for the largest order is the one part of these reports that
costs. It takes no estimate of the omitted potential while the series bound
of the rule settles the matter, as for 3,480 electrons; for 23,768 electrons
none or one (0.12 s), and four at the minimum vacuum; for 39,368 electrons
one (0.17 s) down to 4 ang of vacuum and four to seven below, where the
tolerance halves and order 60 misses it (workstation host). With the shell
sums the block took 0.33 s (one estimate) to 1.2 s (six) for the atoms of
C9449H1572 on 22.6 million points, and 0.16 s with a tolerance line in the
input. `after_scf` of the record holds `estimates`, `shell_seconds`,
`radius_seconds` and `seconds`.

The estimates are a report and no result depends on them. Where the search
finds no radius (a `Domain_Energy_Tolerance` so small that no order meets
the tolerance it implies), the block keeps its estimate and says "No radius
for the wall share: ..."; whatever else fails in a report is printed in its
place as "..., the report failed (...)". Neither stops a calculation nor
loses a finished one.

`PARSEC_DOMAIN_REPORT=0` in the environment (with the switches added beside
it in the last table of switches in `acceleration/README.md`) restores the
report of an input
that gives its radius: no line of the sphere at set-up or after the SCF, no
`domain_*` backend detail, no work after the SCF, and a dry run that says
nothing of the rule; `parsec.out` is then what it was before the rule. For
a radius the rule chose the switch leaves out the block after the SCF only.

`--dry-run`, and `Output_Level` 2 or more, also say what the rule would give
for an input that has a radius, or why it would give none (a tolerance of
the input that no order meets at any radius). Box domains are untouched.
`parsec_python.Hartree.domain.default_domain(atoms, pseudopotentials,
electron_count, spacing=...)` is the rule for callers that build a
`SinglePointInput`, whose radius stays a required number.

What the rule gives for the benchmark nanodiamonds (grid 0.2 ang) against
their inputs with 5 ang of vacuum:

| cluster | electrons | input: radius / order | rule: radius / tolerance / order |
|---|---:|---|---|
| C795H300 | 3,480 | 17.3 ang / 16 | 16.8 ang / 1.0e-3 Ry / 19 |
| C1213H412 | 5,264 | 19.9 / 20 | 19.0 / 1.0e-3 / 24 |
| C1653H508 | 7,120 | 20.9 / 20 | 20.3 / 9.4e-4 / 23 |
| C2455H636 | 10,456 | 24.5 / 25 | 23.8 / 7.8e-4 / 29 |
| C3469H804 | 14,680 | 27.1 / 30 | 26.2 / 6.6e-4 / 36 |
| C4601H988 | 19,392 | 28.0 / 28 | 27.4 / 5.7e-4 / 37 |
| C5659H1132 | 23,768 | 30.7 / 34 | 29.7 / 5.1e-4 / 44 |
| C7067H1308 | 29,576 | 31.6 / 31 | 30.9 / 4.6e-4 / 43 |
| C9449H1572 | 39,368 | 35.1 / 36 | 34.5 / 4.0e-4 / 50 |

Machine-learned densities are optional initial guesses, not new functionals.
They are converted/validated/normalized on the authoritative PARSEC grid and
do not replace the PP core density or any converged DFT term. See
[MLDensity/README.md](MLDensity/README.md) for direct SCDP/ChargE3Net setup,
input labels, cache behavior, and training-domain limitations.

The default mixer remains the strict PARSEC Anderson formula.  Difficult
large systems may opt into a fixed-point-preserving safeguard that limits
exceptionally large Anderson extrapolations and resets stale history after a
large residual increase:

```text
Mixing_Safeguard: true
Mixing_Step_Limit: 2.0
Mixing_Growth_Trigger: 2.0
Mixing_Backoff: 0.5
```

These controls alter only the nonlinear convergence path.  They do not alter
the Kohn--Sham equations, energy functional, or convergence criterion.  Leave
`Mixing_Safeguard` absent or false for a line-by-line PARSEC mixer comparison.

### Energy refinement by a hybrid functional

```text
Energy_Refinement: b3lyp
```

(or `--energy-refinement b3lyp` on the command line, which overrules the
input; `none` switches it off) adds one evaluation after the SCF: the
exchange-correlation energy of the converged orbitals and density is
computed with B3LYP and replaces the one of the self-consistent functional
in the total energy,

```text
E_refined = E_total - E_xc[Correlation_Type] + E_xc[B3LYP].
```

This is the refinement step of Xu, Prendergast and Qian, *J. Chem. Theory
Comput.* **18**, 5471 (2022) ("PP-PBE(B3LYP)" after a PBE SCF; `Lrefine=T`
in ARES). It is not a B3LYP SCF: nothing of the SCF changes, and without the
line a calculation is what it was, to the last bit. `parsec.out` gains a
block "Energy refinement" after the SCF report with the five parts, both
exchange-correlation energies, the refined total and, where the species
have an `Atomic_Energy_Correction`, the refined all-electron total (the
correction is that of the pseudopotentials of the SCF functional). The
result archive gains `refinement_*` members.

The B3LYP energy is that of Gaussian, libxc and ARES:
`0.20 E_x(HF) + 0.08 E_x(Slater) + 0.72 E_x(B88) + 0.19 E_c(VWN-RPA)
+ 0.81 E_c(LYP)`, with the RPA parametrization of VWN. The four density
functionals ([V_xc/b3lyp.py](V_xc/b3lyp.py)) use the gradient of the
self-consistent PBE. The exact exchange
([V_xc/exact_exchange.py](V_xc/exact_exchange.py)) is `-sum_ij f_i f_j
(ij|ij)` over the occupied orbitals: one open-boundary Poisson solve per
pair with the Hartree solver of the calculation, on the full grid, with the
boundary values of the pair density at the order in use and without the
atomic tail, which belongs to the total density. Its cost is
`N (N + 1) / 2` Poisson solves for `N` occupied orbitals; with
`PARSEC_HARTREE_LINEAR_BACKEND=cupy` they run on the GPU like those of the
SCF.

Limits, each stated in the output where it applies: isolated systems only
(a periodic input and molecular dynamics refuse the keyword, as does the
multi-node runner); with fractional occupations the exchange is that of
the ensemble density matrix, of each spin channel in a spin-polarized
calculation and spin-restricted otherwise; with a
nonlinear core correction the density functionals see valence plus core
density and the exact exchange the valence orbitals, a combination that no
reference has checked. The refined energy is not variational and follows
the error of the density to first order. Example, validation against ARES
and timings: [`examples/0_CH4_CF4/python_pbe_b3lyp`](../../examples/0_CH4_CF4/python_pbe_b3lyp/README.md).

## Spin polarization

`Spin_Polarization: true` solves an isolated system with two collinear spin
channels, as PARSEC does: one potential and one eigenproblem per channel,
one Fermi level, and the exchange-correlation functional of the two spin
densities. `Correlation_Type` must be given, `ca` (local spin density,
Perdew-Zunger) or `pbe`; without the line PARSEC uses the Perdew-Wang LSDA,
which is not implemented.

```text
Spin_Polarization: true
Correlation_Type: pbe
States_Num: 12                   # states of each spin channel

Atom_Type: O
Local_Component: p
Initial_Spin_Polarization: 0.3   # of this species; default 0.1
begin Atom_Coord
  0.0  0.0   1.1414
  0.0  0.0  -1.1414
end Atom_Coord
```

`Initial_Spin_Polarization` scales the atomic starting density of a species
by one plus it for the up channel and one minus it for the down channel. The
SCF loop then finds the moment: `parsec.out` reports the electrons of each
channel, the net magnetic moment, the Fermi level and the levels of each
channel at every step, in PARSEC's layout. With `Fermi_Temp` negative,
`occup.in` holds one section per channel, the up channel first, and an
occupation of one is one electron. Forces, relaxation, molecular dynamics and
`Energy_Refinement: b3lyp` work as without spin.

Every execution path runs it: `reference_main.py`, and `main.py` with the
SciPy, native and CuPy backends. The two channels are solved on the full
grid: a GPU run does not use the symmetry representations, and the
multi-node runner refuses a spin-polarized input unless it runs in one
process (`--serial-control`). Periodic cells are refused.

With `ca`, energies, levels and forces agree with Fortran PARSEC to the
digits it prints. With `pbe` they differ as they do without spin, because
this package uses the potential of the discrete energy. A state with an
empty spin channel (a hydrogen atom, any one-electron system) is an ordinary
calculation here; in Fortran PARSEC it does not converge with `ca` and loses
its exchange energy with `pbe`. Two things are not unique in either program:
the sign of the moment, and the orientation of a partly filled degenerate
shell such as the oxygen atom at the default temperature, for which
occupations fixed in `occup.in` give a reproducible state. Details:
[`PYTHON_IMPLEMENTATION.md`](PYTHON_IMPLEMENTATION.md#spin-polarized-calculations)
and [`PARSEC_ALGORITHM.md`](PARSEC_ALGORITHM.md#spin-polarized-calculations).

## Pseudopotentials

The pseudopotential filename is normally `<Atom_Type>_POTRE.DAT`. A
configuration-qualified core-hole type therefore uses, for example,
`C-1s_POTRE.DAT` while `Element_Symbol: C` preserves chemical identity.

An old-format Martins file, `<Atom_Type>_POTR.DAT`, is read with
`Pseudopotential_Format: martins_old`. It holds potentials and radial
functions only, so the input supplies the rest, as for Fortran PARSEC:
`Potential_Num` (the number of channels), the block `Electron_Per_Orbital`
(their occupations, whose sum is the ionic charge) and `Core_Cutoff_Radius`
(the radius of the projectors). The SiNa4 relaxation test of Fortran PARSEC
(`tests/relaxation`) uses one.

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
- [FORCES_AND_RELAXATION.md](FORCES_AND_RELAXATION.md): forces, structural
  relaxation and molecular dynamics: keywords, files, algorithm, differences
  from PARSEC and validation.
- [acceleration/README.md](acceleration/README.md): backend selection,
  optimized kernels, resident execution, and profiling.
- [acceleration/ACCELERATION_AUDIT.md](acceleration/ACCELERATION_AUDIT.md):
  physics-preservation and performance audit.
