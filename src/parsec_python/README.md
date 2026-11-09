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
| `MLDensity/` | Load or predict optional ML initial densities (SCDP, ChargE3Net, custom bridges) on the exact DFT grid or a box; `ml_density.py` is its command line. |
| `Grid/` | Build centered isolated sphere or box grids. |
| `Laplacian/` | Generate high-order finite-difference coefficients and sparse `-nabla^2`. |
| `Pseudopotential/` | Read Martins-new `POTRE.DAT`, old-format `POTR.DAT`, Martins-Wang `POTRW.DAT` and FHI98PP `FHIPP.DAT`, integrate radial data, reproduce PARSEC splines and Fourier filters. |
| `V_ion/` | Assemble local ionic fields, KB nonlocal factors, initial atomic density, and ion-ion energy. |
| `Hartree/` | Construct open-boundary data and solve the finite-difference Poisson equation. |
| `V_xc/` | Evaluate the functional of `Correlation_Type` (CA/PZ, PW92, X-alpha, Wigner, Hedin-Lundqvist, PBE, PW91, LB94, Casida-Salahub, the meta-GGA SCAN), without or with spin polarization, and their energy terms; the grid layer of the meta-GGAs (`tau`, the orbital term, `meta_gga.py`). |
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
- every `Correlation_Type` of Fortran PARSEC: CA/PZ, PW92 (`pl`, the
  default with spin polarization), Slater X-alpha (`xa`), Wigner (`wi`) and
  Hedin-Lundqvist (`hl`) LDA, PBE and PW91 (`pw`) GGA, and the LB94 (`lb`)
  and Casida-Salahub (`cs`, with `Ion_Energy_Diff`) potential corrections of
  CA ([V_xc/selection.py](V_xc/selection.py));
- the functionals of libxc (optional library; LDA, GGA and meta-GGA, their
  hybrids through the Fock loop; on the GPU with a CUDA build), see
  [Functionals of libxc](#functionals-of-libxc);
- meta-GGA functionals of isolated systems and periodic cells (the native
  SCAN, libxc's meta-GGAs such as r2SCAN and TPSS), unpolarized and
  spin-polarized, with forces, on the CPU and on the GPUs of one node, and
  meta-GGA hybrids of isolated systems (libxc's TPSSh, r2SCAN0, SCAN0, ...),
  see [Meta-GGA functionals](#meta-gga-functionals);
- collinear spin polarization of isolated systems (`Spin_Polarization: true`)
  with every functional but `xa`, `wi` and `hl`, which have no spin form, see
  [Spin polarization](#spin-polarization);
- LDA+U of the d channel (`ldaplusu_u`, `ldaplusu_j` per species) and the
  Tkatchenko--Scheffler dispersion correction (`Van_Der_Waals`), see
  [LDA+U and the TS dispersion correction](#ldau-and-the-ts-dispersion-correction);
- self-consistent hybrid functionals of isolated systems (HF, PBE0, PBEh,
  HSE06, HSE03, B3LYP, B3LYP5, B3PW91, LC-wPBE, CAM-B3LYP), unpolarized and
  spin-polarized, with forces, relaxation and molecular dynamics, on the CPU
  and on the GPUs of one node, see
  [Self-consistent hybrid functionals](#self-consistent-hybrid-functionals);
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
  [FORCES_AND_RELAXATION.md](FORCES_AND_RELAXATION.md);
- the same for periodic cells without spin-orbit coupling, at the Gamma point or with k-points (spin-polarized ones at the
  Gamma point): on the readable solver (`main.py` hands such an input to it), for a spin-unpolarized Gamma-point cell on the
  accelerated path, and with k-points on the GPUs of the k-point solver, see "Periodic cells" and "On GPUs" in that document;
- the same for self-consistent spin-orbit coupling (`SO_from_scratch`, alone or with `Spin_Polarization`) and non-collinear
  magnetism, isolated or periodic (Gamma point or k-points), on the readable solver, see "Spin-orbit coupling and
  non-collinear magnetism" in that document.
- With `Use_Fortran_Routines: .true.` (and for the inputs that run on those routines by default - slabs, shifted bulk grids, forces of a bulk cell at the
  Gamma point on a grid both sets share with Cubic_Spline species, bulk cells whose space group sets Fortran's grid -
  or by themselves, see
  `Input/routines.py`), the Fortran-derived routines:
  - the total charge and dipole of an isolated system after every SCF loop
    (`dipole.f90`), the static polarizability from PARSEC's seven finite-field
    SCF loops (`Polarizability`, `Electric_Field`, file `polar.dat`), fixed
    point charges (`Add_Point_Charges`, with their energy and forces) and the
    point-group symmetrization of the density of a cluster
    (`Symmetrize_Charge_Density`); these three run on the reference solver;
  - wire and slab cells (`Boundary_Conditions: wire` / `slab`, PARSEC's 1-D
    and 2-D periodic boundary conditions with `Boundary_Sphere_Radius` the
    radius of the cylinder or half height of the slab, charged sheets of a slab
    with `add_charged_sheets`): Fortran's grid, the Hartree boundary values of
    `hartset_wire`/`hartset_slab`, the Gaussian-compensated local potential,
    the wire ion-ion and slab Ewald energies, Gamma-point and k-point SCF
    (collinear spin too), band structures, forces (also with collinear spin
    and weighted or irreducible k-points), relaxation and molecular dynamics
    (`Relaxation.wire_slab_backend`), Fortran's space group of a wire or slab
    (`alatt` with `4 rmax` along the confined directions; a slab has none
    unless bands, spin-orbit or NSCF turn it on) with `Monkhorst_Pack_Shift`
    and `Symmetrize_Charge_Density`, `Polarizability`, parsec.dat with
    `Restart_Run` and `nscf_kpoints`, `Super_Cell`/`Super_Cell_Vac`,
    `Create_DOS` and LDA+U (at Gamma, with k-points, spin and bands; the
    Bloch projectors replicate along the periodic directions), on the reference solver
    (`SCF/wire_slab.py`).
  - PARSEC's k-point machinery for periodic cells: with a `Monkhorst_Pack_Shift`
    block the Monkhorst-Pack grid is Fortran's `((i-1)+s)/n`, reduced to the
    irreducible wedge by the space group (found as `symmetries.f90` does) and
    time reversal, with PARSEC's tables and `qpoints.dat`; the k-points are then
    folded into `(-1/2, 1/2]` of the reciprocal lattice (PARSEC 1.5 keeps them in
    `[0, 1)`; the finite-difference Bloch operator is not invariant under
    `k -> k + G`, so the folded set makes a run without symmetry agree with the
    symmetric one; Andrew Grieder's routines fold their shifted grid alike), and the density is
    symmetrized over the space group (`Symmetrize_Charge_Density`, then true by
    default); weighted `Kpoint_List` rows; reference solver only.  Without the
    shift block the unreduced grid of earlier versions is kept.  On Andrew
    Grieder's routines (`Use_Fortran_Routines: .false.`) the shifted grid of a
    bulk single point is reduced by the operations that map their grid (that of
    `Grid_Spacing`) and the mesh onto themselves, with time reversal, and the
    density averaged with them (`Grid.space_group.find_grid_space_group`,
    `Grid.kpoints.mesh_preserving_operations`; README "Irreducible k-points on
    the default routines").
  - in every SCF driver (cluster, Gamma, k-points, periodic spin, spin-orbit,
    non-collinear): `Dynamic_Diag_Tol` (off with `Restart_Run`, as PARSEC),
    `Save_Intermediate_Charge_Density` / `Save_Intermediate_Eigenvalues`
    (`parsec.dat` and `eigen.dat` after every iteration),
    `Fix_Number_Eigenvalues` with `Restart_Run` (the eigensolver keeps the
    stored `States_Num` states, no buffer); `nscf_kpoints` of a spin-orbit run
    (from its spinor `parsec.dat`, `Spin_z` of every state); PARSEC's symmetry
    report (`symchk`, Abelian subgroup and character table, the
    multiplication tables at `Output_Level >= 2`) and ion-ion force table for
    every run (`Grid/symmetry_analysis.py`; report only, the fields stay on the
    full grid);
  - forces, relaxation and molecular dynamics of periodic cells at Gamma and
    with k-points (also collinear spin, an irreducible wedge completed over the
    space group, `Double_Grid_Order > 1`) and of the spin-orbit
    (`SO_from_scratch`) and non-collinear drivers, isolated or periodic, on
    the reference solver; on a GPU the accelerated command line runs those of
    an orthorhombic bulk cell at Gamma (no spin) on its own path and builds
    the ionic fields of every orthorhombic bulk cell whose atoms move, and
    their forces, on the device
    ([FORCES_AND_RELAXATION.md](FORCES_AND_RELAXATION.md#periodic-cells)).

Not currently supported as production physics:

- forces, relaxation and molecular dynamics of the perturbative spin-orbit correction (`SO_PSP` without `SO_from_scratch`), of
  spin-polarized periodic cells with k-points, of spin-polarized periodic cells and spin-orbit or non-collinear inputs on
  the accelerated path (`main.py` hands them to the readable solver), and of periodic cells and non-collinear inputs on
  several nodes (`--mpi`);
- hybrid functionals of wires and of periodic cells on several nodes (`--mpi`), Energy_Refinement
  on several nodes, meta-GGA functionals on several nodes (`--mpi`), meta-GGA hybrids of periodic cells,
  meta-GGAs with spin-orbit coupling, non-collinear magnetism or `Restart_Run`, libxc functionals with
  `--mpi` (hybrids included), the TS dispersion correction or an empirical dispersion term of their own
  (-D, -D3), and DFT+U beyond PARSEC's LDA+U of the d channel (other channels or forms);
- stress (no stress tensor is computed for any functional);
- PARSEC restart files other than `relax_restart.dat` and `mdinit.dat`;
- Ono--Hirose double-grid order greater than one.
- With `Use_Fortran_Routines: .true.` (and for the inputs that run on those routines by default - slabs, shifted bulk grids, forces of a bulk cell at the
  Gamma point on a grid both sets share with Cubic_Spline species, bulk cells whose space group sets Fortran's grid -
  or by themselves, see
  `Input/routines.py`), the Fortran-derived routines:
  - forces with the perturbative spin-orbit correction (`SO_PSP` without
    `SO_from_scratch`);
  - PARSEC restart files other than `relax_restart.dat` and `mdinit.dat`.

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
[the ML initial-density tutorial](MLDensity/README.md) for installing the
models, `ml_density.py`, input labels, connecting other models, cache
behavior, and training-domain limitations.

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

PARSEC's other mixers are available with its keywords and defaults:
`Mixing_Method: broyden` (the generalized Broyden mixer of `genbro.f90`,
with `Mixing_Param`, `Memory_Param` and `Restart_Mixing`) and
`Mixing_Method: msecant1`, `msecant2` or `msecant3` (the Type-I, Type-II and
hybrid multisecant mixers of `msecant1-3.f90`, with `Mixing_Group_Size`,
`Mixing_EN_Like`, `Mixing_Restart_Factor`, `Mixing_Memory_Expand_Factor` and
`Mixing_Preferred_Type`).  The safeguard belongs to the Anderson mixer only.

### Bulk cells: the SUBSPACE lower bound

The later-SCF Chebyshev filter (SUBSPACE) damps the unwanted interval `[l, u]`.  PARSEC's rule, this code's
default, is `l = max(r_max, 0)` on the first call after the first solve and `l = max(l_previous, r_max)` later
(`r_max` the largest saved Ritz value).  The clip at 0 suits clusters, whose unwanted spectrum starts near the
vacuum level.  A 3-D periodic cell has none: with this code's periodic local potential the eigenvalues of bulk
diamond lie near -37 Ry (64 atoms) and -84 Ry (216 atoms), the offset growing with the cell, so `l` stays 0 for
the whole SCF and the conduction band between the subspace and 0 Ry is amplified with the occupied states.  The
orbitals then converge by a few per cent per SCF iteration, the potential residual (the SCF test) becomes small
because the orbitals change little, and the run stops with the energy far above its converged value.  PARSEC 1.5's
`Smart_Subspace: .true.` takes `l = r_max` wherever the spectrum lies (`subspace_filter_bounds` in
`Eigensolvers/subspace.py`, the one rule of the reference filter and of the device filters of one GPU, of a
distributed basis and of the k-point solver).  Bulk diamond at the Gamma point (the inputs of
`examples/3d_diamond_benchmark`: 18 points per cubic edge, order 8, chebff, the nanodiamond blocks, 2e-4 Ry; one
A100):

| atoms | default: iterations / SCF s / E - E_converged | `Smart_Subspace: .true.` |
|---|---|---|
| 64 | 24 / 1.6 / +0.062 mRy/atom | 8 / 1.2 / +0.00001 mRy/atom |
| 216 | 43 / 12.6 / +0.37 mRy/atom | 10 / 3.7 / +0.00002 mRy/atom |
| 512 | 65 / 118 / +4.9 mRy/atom | 16 / 19.2 / +0.00001 mRy/atom |
| 1,000 | 104 / 954 / +13.9 mRy/atom | 17 / 85.5 / +0.0015 mRy/atom (`Chebyshev_Degree: 20`: 17 / 89.1 / +0.000004) |

(`E_converged`: `Smart_Subspace`, `Chebyshev_Degree: 40` without the degree decrease, a buffer of 10 % of the
states, `Convergence_Criterion: 1e-6 Ry`.)  With the smart bound the other blocks matter little: degree 20 or 30,
no degree decrease, buffers of 5 or 10 % and `FF_MaxIter` 4 or 6 change the 512-atom SCF by -6 to +3 iterations
and its time by less than 15 %, all within 0.001 mRy/atom; at 1,000 atoms degree 20 instead of 15 brings the
stopping energy from 0.0015 to 0.000004 mRy/atom above the converged one for 4 % more time.  With the smart bound
`Mixing_Param: 0.6` stops at +0.0001 mRy/atom (512 atoms); with PARSEC's rule it stopped 0.7 Ry high (216 atoms).  The same holds for k-points (64 atoms, 2x2x2 mesh:
17 iterations and 0.048 mRy/atom high with PARSEC's rule, 8 with the smart one) and for bulk Si (64 atoms, PBE,
1e-6 Ry: 51 iterations against 23).  A slab keeps a level near 0 (h-BN, 6x6x1: 25 iterations with PARSEC's rule, 30
with the smart one, the same energy), and so do clusters (Si28H36: 12 and 13).  Since 8 November 2026 a bulk cell
(`Boundary_Conditions: bulk`) without a `Smart_Subspace` line takes the smart rule; slabs and wires keep PARSEC's
rule (slabs measured slower, wires not measured), and clusters take the input's rule as before
(`DEFAULT_PERIODIC_SUBSPACE_LOWER_BOUND` of `Input/parsec_input.py` holds the rule of each periodic boundary
condition for inputs without the line).  `Subspace_Lower_Bound: monotone | smart | adaptive` (PARSEC.py) names
each rule; `Smart_Subspace: .true.` is `smart` and `.false.` PARSEC 1.5's `adaptive` (not the earlier PARSEC's
`monotone`), as in 1.5.

### The upper bound of the filters, and the other PARSEC 1.5 defaults

Every Chebyshev filter (CHEBFF, CHEBDAV, SUBSPACE) takes its upper end `u` from a short Lanczos run
(`lanczos_bound_from_ritz` in `Eigensolvers/spectral_bounds.py`, the one formula of the reference and device
filters).  This code's default is the earlier PARSEC's radius `u = max(theta_max, |theta_min|) + margin` (4 to 8
steps); PARSEC 1.5's default, `Bounds_Method: lanczos`, is `u = theta_max + margin` with `max(degree/2,
Bounds_Iter = 10)` steps, and moves with the spectrum.  Both run on the reference and the device filters since 8
November 2026 (`Bounds_Method: lanczos_full` stays with the reference CHEBFF).  With a periodic potential whose mean
lies tens of Ry below 0 the radius takes `|theta_min|` and widens the filter interval several times (bulk diamond,
Fermi level -341 Ry at 1,728 atoms: 25 SCF iterations with the radius, 11 with 1.5's bound, 16 ranks); with the
zero-mean convention (`Periodic_Local_Potential: g_space`, `Zero_Average_Vxc: .true.`) the two bounds agree (512 and
1,000 atoms: 10 iterations with the radius, 11 with 1.5's bound), and the radius stays the default.  In the same
convention the other defaults where PARSEC 1.5 differs were measured on bulk diamond of 512 and 1,000 atoms and stay
as they are: 1.5's degree decrease below `2 x Convergence_Criterion` instead of `100 x` (the same iterations and
time), 1.5's working space without buffer (`Exact_State_Count: .true.`, the same or one iteration more), more cycles
of the first filter (`FF_MaxIter` 6 or 10: 7 iterations instead of 10 but 25-90 % more time), the three lower-bound
rules (`Subspace_Lower_Bound`: 10 iterations each at 216 and 512 atoms, smart 10 and the others 11 at 1,000; a
spin-polarized SCAN Si2 cell with two k-points: 25 each, where the smart rule needed 54 against 28 with the offset).
Bulk cells keep the smart rule.

### Kerker preconditioner of the mixing (periodic cells)

`Mixing_Preconditioner: kerker` (PARSEC.py; Fortran PARSEC has no preconditioner of its mixers) filters the
potential residual `R = V_out - V_in` of a 3-D periodic cell in G-space before the mixer of `Mixing_Method` sees it
(`Mixer/preconditioner.py`):

```text
R -> P R,   P(G) = (G^2 + q0^2/eps)/(G^2 + q0^2) for G != 0,   P(0) = 1
Mixing_Preconditioner: kerker        # auto (default) = none here; none
Kerker_Wavevector: 1.0               # q0, bohr^-1 (or ang^-1), default 1.0
Kerker_Dielectric_Constant: 5.7      # eps, default infinite (Kerker's G^2/(G^2 + q0^2))
```

The linear response of the output potential gives `R = -eps e` for an input error `e` (`eps = 1 - (v_H + f_xc)
chi0` acting on potentials), so the ideal step is `eps^-1 R`; Kerker's `P` is `eps^-1` of Thomas-Fermi screening
(a metal), the finite `eps` the model of an insulator whose `eps(G -> 0)` is its dielectric constant.  The mean of
`V_in` is a gauge of the periodic cell (it shifts every eigenvalue and changes neither the density nor the energy),
so the mean of the residual passes unchanged.  The mixer works on the preconditioned fixed-point problem `P R = 0`
(left preconditioning; `P` is invertible, so the converged potential is that of the run without it), and the SCF
test still reads `R` itself.  Two spin channels have the charge part of their residual filtered.  Oblique cells go
through the reciprocal metric.  Clusters, wires and slabs are refused, as are meta-GGAs, hybrids, spin-orbit
coupling, non-collinear magnetism and FDET (untested there).  The FFT runs with NumPy on the host fields of the
mixer (CuPy for device arrays).  Measured on bulk diamond (one A100): with PARSEC's SUBSPACE bound the energy at
iteration k is the same for every mixer and the iteration count follows the eigensolver; with `Smart_Subspace` the
plain Anderson mixer takes 8, 10 and 16 iterations for 64, 216 and 512 atoms, Kerker's form (q0 = 1 bohr^-1) 10, 14
and 27 (it damps the long wavelengths of an insulator too much), the dielectric form (q0 = 1, eps = 5.7) 9, 9 and 11
(512 atoms: 16.6 s instead of 19.2 s, but it stops 0.0014 mRy/atom above the converged energy, the plain mixer
0.00001).  The preconditioner therefore stays opt-in.

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
`N (N + 1) / 2` Poisson solves for `N` occupied orbitals; they run where
the conjugate gradients of the SCF do, on the GPU of a GPU run (the default
since 2026-11-07; `PARSEC_HARTREE_LINEAR_BACKEND=native` keeps both on the
host, see [acceleration/Hartree/GPU_HARTREE.md](acceleration/Hartree/GPU_HARTREE.md)).

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
With a hybrid `Correlation_Type` the refinement is refused (the functional is
then self-consistent, see the next section).

## Self-consistent hybrid functionals

```text
Correlation_Type: pbe0
```

runs a generalized Kohn-Sham SCF with a nonlocal exact-exchange operator for
an isolated system (`Boundary_Conditions: cluster`, sphere or box domain) or a
periodic cell (`bulk` at the Gamma point or with k-points, `slab`; see
[Periodic cells](#periodic-cells)), spin-unpolarized or with collinear spin
(`Spin_Polarization: true`, also with the occupations of `occup.in`), with
forces, structural relaxation and molecular dynamics. The functional is

```text
E_xc = a_x E_x^HF[K] + E_xc^semilocal[n],
```

the exact (Hartree-Fock) exchange of the occupied valence orbitals with the
kernel `K` of the functional (`1/r`, the short-range `erfc(omega r)/r` or the
long-range `erf(omega r)/r`) and a semilocal part with the reduced semilocal
exchange of the functional. Every orbital sees the Hamiltonian
`-nabla^2 + V_ion + V_H + V_xc^semilocal + V_NL + a_x V_x`, where `V_x` is the
exchange operator of the occupied orbitals of the same spin
([V_xc/exchange_operator.py](V_xc/exchange_operator.py),
[SCF/hybrid.py](SCF/hybrid.py), [V_xc/hybrid.py](V_xc/hybrid.py)).

### Functionals

| label | functional | exact exchange | semilocal part |
|---|---|---|---|
| `hf` | Hartree-Fock | `1/r`, 1 | none |
| `pbe0` | PBE0 | `1/r`, 0.25 | 0.75 PBE exchange + PBE correlation |
| `pbeh` | PBEh(a) | `1/r`, a (`Exact_Exchange_Fraction`, default 0.25) | (1 - a) PBE exchange + PBE correlation |
| `hse06` | HSE06 | `erfc(omega r)/r`, 0.25, omega = 0.11 bohr^-1 | PBE exchange - 0.25 wPBE short-range exchange (omega = 0.11) + PBE correlation |
| `hse03` | HSE03 | `erfc(omega r)/r`, 0.25, omega = 0.15/sqrt(2) = 0.1061 bohr^-1 | PBE exchange - 0.25 wPBE short-range exchange (omega = 0.15 * 2^(1/3) = 0.1890) + PBE correlation |
| `b3lyp` | B3LYP (VWN-RPA, as Gaussian and libxc) | `1/r`, 0.20 | 0.08 Slater + 0.72 B88 + 0.19 VWN-RPA + 0.81 LYP |
| `b3lyp5` | B3LYP with VWN5 (Quantum ESPRESSO's `b3lyp`) | `1/r`, 0.20 | 0.08 Slater + 0.72 B88 + 0.19 VWN5 + 0.81 LYP |
| `b3pw91` | B3PW91 | `1/r`, 0.20 | 0.08 Slater + 0.72 B88 + 0.19 PW92 + 0.81 PW91 correlation |
| `lc-wpbe` | LC-wPBE | `erf(omega r)/r`, 1, omega = 0.4 bohr^-1 | wPBE short-range exchange (omega = 0.4) + PBE correlation |
| `cam-b3lyp` | CAM-B3LYP | `1/r`, 0.19, + `erf(omega r)/r`, 0.46, omega = 0.33 bohr^-1 | 0.35 B88 + 0.46 short-range B88 (omega = 0.33) + 0.19 VWN5 + 0.81 LYP |

B88 includes its Slater part. The short-range wPBE exchange is Heyd's
exchange-hole form, the one of Quantum ESPRESSO and VASP; HSE06 uses one
omega for both parts. Case, `-` and `_` do not matter in a label, and the
spellings `hartree-fock`, `pbe1pbe`, `hse`, `hseh1pbe`, `hse2pbe`, `lcwpbe`
and `camb3lyp` are accepted. Each functional of the registry states its
references, its libxc counterpart and how libxc, Gaussian and Quantum
ESPRESSO differ (`HybridFunctional.variants`).

### Keywords

| keyword | default | meaning |
|---|---|---|
| `Exact_Exchange_Fraction` | that of the functional | `a` of `pbeh`, and the short-range fraction of `hse06` and `hse03` (HSE(a)); in (0, 1]; refused for every other functional, which is defined by its fraction (`pbe0` asks for `pbeh`) |
| `Screening_Parameter` | that of the functional | omega in bohr^-1 (a number, or with `bohr^-1` / `ang^-1`) of `hse06`, `lc-wpbe` and `cam-b3lyp`, in the exact and the semilocal part together; refused for `hse03` (defined by two omegas) and for the global hybrids |
| `Exx_Max_Iter` | 100 | outer (Fock) iterations, a positive integer; Quantum ESPRESSO's outer loop has the same cap |
| `Exx_Tolerance` | `1e-6 Ry` | the estimated exchange error below which the Fock loop stops (an energy, in the units of the other energy keywords) |
| `Exx_Batch_Size` | 32 | pair densities per batched Poisson solve, a positive integer |
| `Exx_Pair_Solver` | `kronecker` | `kronecker` (every kernel) or `cg` (the conjugate gradients of the Hartree solver, `1/r` only) |
| `Exx_Inner_Criterion` | `adaptive` | `adaptive`: while the Fock loop is far from converged an inner SCF stops at a share of its first residual (a tenth of the next loop's expected first residual), the two last loops at `Convergence_Criterion` and with every one of the `States_Num` levels converged at their last Hamiltonian, a spin-unpolarized PBE start with integer occupations at `max(Convergence_Criterion, 1e-4 Ry)`; `fixed`: every inner SCF and the start at `Convergence_Criterion` |
| `Exx_Mixer_History` | `restart` | `restart`: a new mixer per inner loop, as PARSEC starts every SCF loop; `carry`: the inner loops continue one Anderson history (the first secant pairs of each loop carried into the next, the first loop seeded by the PBE start), measured slower with the adaptive criterion (opt-in). The Broyden and multisecant `Mixing_Method`s always restart |
| `Exx_Outer_Mixing` | `diis` | `diis`: Anderson (Pulay) extrapolation of the density matrix of the outer loop, through natural orbitals, where the loop is slow; `none`: the plain fixed point |
| `Exx_Divergence` | `auto` | periodic cells: the treatment of the Coulomb singularity at `q + G = 0`, `auto`, `gygi-baldereschi`, `gygi-baldereschi-x` or `none` ([Periodic cells](#periodic-cells)) |
| `Exx_Auxiliary_Exponent` | the limit `a -> 0` (`10/E_w` for a truncated slab) | periodic cells: the exponent (bohr^2) of the Gygi-Baldereschi auxiliary function, zero or positive |
| `Exx_Slab_Coulomb` | `padded` | slabs: `padded` (the bulk kernel on the slab box padded to twice its height), `truncated` (the Coulomb kernel cut along z) or `periodic` (the bulk kernel on the slab box itself, as Quantum ESPRESSO's exact exchange) |

The twelve keywords without a hybrid `Correlation_Type` are refused.
`Convergence_Criterion`, `Max_Iter`, the mixer and the eigensolver settings
apply to every SCF loop of the Fock loop. `Generate_Complete_Parsec_In`
writes the Fock-loop keywords with their values (the fraction and omega where
the input sets them).

### Method

1. **Start.** An SCF with PBE (for every hybrid), or, after an ionic move,
   the orbitals of the structure before. The potential is then rebuilt from
   the converged density with the semilocal part of the hybrid, and the
   eigensolver keeps its subspace.
2. **Operator.** `W = V_x Psi` for the `States_Num` orbitals of each spin
   channel (the occupied ones and the empty states the input asks for, so
   the LUMO is in the span), each pair of occupied orbitals solved once, and
   its adaptively compressed form (ACE, Lin 2016; the batched pair solves
   follow the principles of SPARC, Jing, Sharma, Pask and Suryanarayana,
   arXiv:2501.16572 and arXiv:2406.16998):
   `M = -Psi^T W = L L^T`, `xi = W L^-T`, `V_ACE = -xi xi^T`, equal to `V_x` on
   the span of `Psi` and applied as two matrix products.
3. **Inner SCF.** The SCF loop of the code with `a_x V_ACE` fixed in every
   Hamiltonian: the Chebyshev filters, the Lanczos bounds, the Rayleigh-Ritz
   step and the residuals. Its iterations report the energy of the
   functional with the exchange linearized about the orbitals of the
   operator (`(linearized: fixed operator)` in `parsec.out`).
4. **Stop test.** `E_x` of the new orbitals `P` and Quantum ESPRESSO's
   estimated exchange error
   `dexx = a_x [2 E_x(P, P0) - E_x(P0) - E_x(P)] >= 0`, `P0` the orbitals of
   the operator. The loop stops when `dexx < Exx_Tolerance` and the inner
   loop converged; otherwise the operator of `P` is built and the next inner
   loop starts from the last one. The last outer iteration evaluates only
   the occupied pairs.
5. **Fock-loop controls** (where the loop goes is set by `Exx_Tolerance`
   and `Convergence_Criterion`; these keywords set how it gets there, see
   [SCF/fock_controls.py](SCF/fock_controls.py) and
   [SCF/fock_diis.py](SCF/fock_diis.py)).
   *Adaptive inner criterion*: the inner loop `k` only has to be accurate
   against the change the next operator brings, which shows as the first
   residual of the next loop and falls by a factor `rho` per outer iteration
   (0.06-0.25 for 20-25 % exact exchange, 0.25-0.47 for HF on the 49 cases of
   `examples/0d_hybrid_functionals`); it stops at
   `max(Convergence_Criterion, 0.1 rho r_k)`, `r_k` its own first residual
   (an inexact Newton forcing term; Quantum ESPRESSO's `adaptive_thr` does the
   same with its energy threshold). A loop whose dexx is expected below thirty
   times `Exx_Tolerance` runs at `Convergence_Criterion`, and a loop that
   ended loose never ends the Fock loop: the final state is always an inner
   loop at `Convergence_Criterion` with `dexx < Exx_Tolerance`. Fewer SCF
   iterations filter the empty states less, so a loop run at
   `Convergence_Criterion` (with this criterion or with the outer DIIS) ends
   with one more iteration on its last `V_in`: eigensolver passes until no
   one of the `States_Num` levels moves by more than `Exx_Tolerance`, its
   density and energies from those eigenpairs (no pair solves; `levels
   converged by n eigensolver passes` in `parsec.out`). A spin-unpolarized
   PBE start stops at `max(Convergence_Criterion, 1e-4 Ry)` (QE's
   `conv_thr_init`) when its occupations are integer (largest `min(f, 1 - f)`
   below 1e-3): the first inner loop starts from a residual of 3e-3..4e-2 Ry,
   and the start's energy, within ~1e-6 Ry of the converged PBE energy, is a
   start (run PBE for that energy). A spin-polarized start, and an
   unpolarized one with a partly filled level (which then goes on from where
   it stopped), run to `Convergence_Criterion` on any grid: an open shell can
   keep a soft rotation among degenerate partners (the O atom's occupied p
   orbital on the full grid; in the sectors d_z2 and d_x2-y2 share a
   representation of D2h, p partners one of Cs, C2 or Ci), which an
   unfinished start leaves turned and the Fock loop then holds (1.1e-7 Ry
   above the converged state at `Exx_Tolerance` 1e-8).
   *Carried mixer* (`Exx_Mixer_History: carry`, opt-in): the operator
   changes the inner SCF map mostly by an offset, not its Jacobian, so the
   secant pairs `(dV, dF)` of one inner loop stay valid in the next (absolute
   pairs do not); the first secant pairs of each loop enter the Anderson
   least squares of the next one. On the 49 hybrid examples it cost more
   inner iterations than a fresh mixer per loop (with the adaptive
   criterion: N2 293 against 242, H2O 291 against 231), hence the default
   `restart`.
   *Outer DIIS*: `X_{n+1} = sum_k c_k Y_k`, `c = argmin ||sum c_k (Y_k - X_k)||`
   (Frobenius, from the overlaps of the stored orbitals), `sum c_k = 1`, with
   `X_k` the density matrix of the operator of loop `k` and `Y_k` that of its
   orbitals; the operator of `X_{n+1}` is built from its natural orbitals with
   fractional occupations (the `n_occ` largest; the empty states of the last
   loop complete the ACE) by the same pair solves. It is used only when the
   last orbitals have integer occupations (a symmetric half-filled ensemble
   is a saddle that an extrapolation is drawn to), dexx did not rise in the
   last step, the plain loop converges slowly (dexx ratio above 0.05) and the
   next loop is neither expected to be the last nor the last allowed
   (`Exx_Max_Iter`); a step that raises dexx or the energy is rejected and
   taken back (its output is dropped and the next operator is that of the
   output the step started from). The last operator is always the plain ACE
   of the last orbitals, so the stop test is unchanged. The window keeps
   copies of the occupied orbitals of up to four loops (each sector on its
   device), and only while that gate is open (integer occupations, no rise,
   a slow loop, the end not in sight): a run that never extrapolates holds no
   copy, and the first extrapolation of a slow phase follows two loops after
   the output that opened the gate (two residuals take three outputs). The
   copies are not in the device plan of the
   exchange that the memory rules of the sectors see: a copy is made only
   where the device has the copy, the build of the extrapolated operator
   (three sets of `States_Num` columns) and a tenth of its memory free, and
   a copy or a build that does not fit stops DIIS (plain steps, said in
   `parsec.out`). `parsec.out` prints an `Inner SCF:` line (the criterion
   used, the carried pairs, the level passes) and an `Outer step:` line per
   outer iteration.

The total energy after each inner loop is that of the hybrid functional,
`E_formula - 2 a_x E_x(P, P0) + a_x E_x(P)`: the band energy holds
`2 a_x E_x(P, P0)` of exchange, which is removed (double counting), and the
exact exchange of the new orbitals is added. With fractional occupations the
exchange is that of the ensemble density matrix of each spin channel.
`parsec.out` reports every outer iteration (`E_x`, the change, dexx, the
total energy, the pair solves) and a final block with the energy terms of
the generalized Kohn-Sham functional. The result archive holds
`energy_semilocal_exchange_correlation_ry`, `energy_exact_exchange_ry` and
`energy_exact_exchange_band_ry`, the `exx_*` members (fraction, `E_x`,
dexx, the history of the outer iterations with `exx_history_columns`) and
`scf_history_loop`, the loop of every row of `scf_history` (0 for the PBE
start, `n` for the inner loop of outer iteration `n`).

**Pair solver.** `Exx_Pair_Solver: kronecker` solves the finite-difference
Poisson problem of the run's Laplacian directly on the bounding box of the
domain ([Hartree/kronecker.py](Hartree/kronecker.py)): the three 1-D
Dirichlet matrices are diagonalized once, and a batch of pair densities
costs three batched matrix products, a Hadamard product and three products
back (`12 n^4` flops for `n` points per axis). The boundary values beyond
the box are the multipole expansion of each pair density at the order the
Hartree boundary of the run uses, folded into the right-hand side. The
expansion converges only beyond the farthest point that carries density,
so the solver pads a box, ellipsoid or cylinder domain automatically until
the exterior point nearest to the origin lies beyond the farthest active
point (a sphere needs no padding). The screened kernels are the Coulomb
solution smoothed by a Gaussian (`erf`) plus the smoothed exterior tails,
and `erfc = 1/r - erf`; a two-kernel functional (CAM-B3LYP) is one solve of
the weighted kernel. `Exx_Pair_Solver: cg` uses the conjugate gradients of
the Hartree solver on the domain itself, with the multipole (sphere) or
direct (box) boundary of each pair density and the boundary harmonics
formed once.

### Command lines and GPUs

- `reference_main.py` runs the Fock loop on the host, with the host
  Kronecker solver: single points, forces, relaxations and molecular
  dynamics.
- `main.py` with the CuPy backend runs the same loop on the GPUs of one
  node: `a_x V_ACE` lives in the device Hamiltonians, of the full grid or of
  every symmetry sector (`-xi_G xi_G^T` per sector), and the pair densities
  of all sectors are solved by the device Kronecker solver on every GPU of
  the process, in tiles of `Exx_Batch_Size` columns, each pair once. Its
  records, energies and stop test are those of the reference loop. A
  spin-polarized hybrid runs both channels in the sectors by default
  (`Spin_Symmetry_Sectors: auto`, with the symmetry guard of
  [Spin channels in the symmetry sectors](#spin-channels-in-the-symmetry-sectors)).
- Orbitals of symmetry sectors are never unfolded to the full grid for the
  pair densities: where the Kronecker box is symmetric about the origin
  (every cluster domain with the default grid shift 1/2 or shift 0), a pair
  density is split into its reflection-parity classes `(t_x, t_y, t_z)`,
  formed directly from the sector data, and each class is solved on the
  octant of the box with the half-size even or odd blocks of the
  one-dimensional Laplacian (`Hartree/kronecker.py`, "Reflection-parity
  classes"). A group of `g` diagonal reflections (D2h 8, D2/C2v 4, Cs/Ci/C2 2)
  leaves `8/g` classes per pair, each 1/16 of the flops of the full box: on
  the nanodiamonds (D2) the ACE builds are about 5x faster. The result is
  that of the full-box solve to round-off; `PARSEC_CUPY_EXCHANGE_PARITY=off`
  takes the full-grid path instead (comparisons), as do orbitals on the full
  grid and groups of one diagonal reflection. The two channels of a
  spin-polarized run in the sectors take the parity classes as well.
- The device bytes of the exchange (`xi` of every channel and sector, the
  sector expansions, the Kronecker plans and the tile work) are counted by
  the memory rules of the sectors, which spill sector states to the host
  where the states and the exchange do not fit together. A build whose
  `W = V_x Psi` cannot be held where its sectors are is refused with the
  numbers (use more GPUs, fewer `States_Num` or a smaller `Exx_Batch_Size`).
- On several nodes the MPI runner (`scripts/run_multi_gpu_scf.sh`, one rank
  per node) runs the same loop: its ranks divide the symmetry sectors (of
  both channels of a spin-polarized run), every sector basis on one GPU,
  and the pair densities of all sectors are solved on every GPU of every
  rank; the orbitals of the other ranks' sectors reach a rank by broadcasts
  and the parts of `W` go back to the owner of their sector, as device
  arrays of Cray MPICH. The decisions of the loop are taken on the root, so
  the runs of one and of several nodes take the same SCF and outer
  iterations and agree within the summation round-off of the exchange
  (`E_x` to 3e-11 Ry; total energies 1e-13 to 1.3e-9 Ry, the larger ones
  where the inner SCF stops at 2e-4 Ry). Single points, forces, relaxations and molecular dynamics (the
  sector of the identity on one GPU of the root, its exchange on all GPUs
  of all ranks) run there, those of a spin-polarized run single points
  only; see
  [acceleration/MULTI_GPU.md](acceleration/MULTI_GPU.md#hybrid-functionals-on-several-nodes).
- `main.py` hands any other hybrid input to the reference solver with a
  NOTE: a CPU backend, or `Exx_Pair_Solver: cg`. `--backend cupy` or
  `native` and `--device gpu` refuse such an input, and so does `main.py
  --mpi` (the k-point solver, which divides the rows of every orbital).

### Periodic cells

A bulk cell (`Boundary_Conditions: bulk`) at the Gamma point or with a
Monkhorst-Pack mesh (`Monkhorst_Pack_Grid`, with or without
`Monkhorst_Pack_Shift`, or a `Kpoint_List` that writes out every point of a
uniform mesh with equal weights) and a slab (`Boundary_Conditions: slab`, the
mesh in its plane) run the same Fock loop, on either set of routines
(`Use_Fortran_Routines`) and on both command lines, with
[SCF/hybrid_periodic.py](SCF/hybrid_periodic.py) and
[V_xc/bloch_exchange.py](V_xc/bloch_exchange.py). On the GPUs of one node
(`main.py` with CuPy) the Gamma point without spin runs on the accelerated
driver's device loop with a cuFFT pair solver, and k-points (and spin) run
the loop of the readable solver with its k-point eigenproblems, the pair
solves, ACE and the exchange term on the GPUs; both give the numbers of the
host loop to round-off ([acceleration/README.md](acceleration/README.md),
"Hybrid functionals of periodic cells on the device"):

- **k-points.** The exchange operator sums over the full mesh, so the space
  group does not reduce it: a hybrid takes the mesh reduced by time reversal
  only (`qpoints.dat` and the report list those points). The SCF solves the
  representatives `k` (weight `2/N`, or `1/N` where `-k = k` modulo a
  reciprocal lattice vector); the factor at `-k` is `conj(u_k)`, exact for the
  finite-difference Bloch operator (`H_{-k} = conj(H_k)`).
- **Pair potentials.** For a Bloch factor `u_nk` and an occupied `u_mk'` the
  pair density `conj(u_mk') u_nk / dV` is the periodic part of a Bloch
  function of `q = k - k'`; its potential is `w = ifftn(K(q + G) fftn(rho))`
  on the box of the grid (`Hartree/exchange_kernel.py`: `K` of
  `alpha/r + beta erfc(omega r)/r`, the CAM coefficients of the functional), and
  `V_x u_nk = -(1/N) sum_k' sum_m f_mk' u_mk' w`. `E_x = (s/2) sum_k w_k sum_n f_nk <u_nk|V_x u_nk>`.
  A pair of two occupied states is solved once for both (its mirror has
  the conjugate potential, or the same one for a time-reversal partner).
  The batched FFTs use half the CPUs of the process (`PARSEC_FFT_WORKERS` sets
  the number).
- **The Coulomb singularity** at `q + G = 0` (`Exx_Divergence`, default
  `auto`): Gygi-Baldereschi with the extrapolation of Nguyen and de Gironcoli
  on the `1/r` part of a bulk cell; `gygi-baldereschi` and
  `gygi-baldereschi-x` are Quantum ESPRESSO's treatments with
  `x_gamma_extrapolation` false and true (its default), `none` leaves the
  Coulomb element out. `Exx_Auxiliary_Exponent` (bohr^2) is the exponent of
  the auxiliary function, by default the limit `a -> 0` (closed-form lattice
  sums); Quantum ESPRESSO uses `10/ecutwfc` (Ry), which a like-for-like
  comparison passes. It matters only on coarse meshes. A long-range
  corrected functional (`lc-wpbe`, CAM-B3LYP) takes the extrapolation on
  every part of its kernel where the mesh does not resolve omega.
- **Slabs.** By default (`Exx_Slab_Coulomb: padded`) the pair densities are
  zero-padded to twice the height of the slab box and solved with the bulk
  kernel of that box (the Gygi-Baldereschi element with the extrapolation on
  its `1/r` part): the images along z sit beyond the density and the
  in-plane mesh converges as fast as in a bulk cell. `truncated` cuts the
  Coulomb kernel along z on the same padded box (exactly isolated, but its
  error falls only as `N_k^-1.5` with the in-plane mesh, a cusp at
  `Q_par = 0` that no element at `q + G = 0` removes); `periodic` keeps the
  bulk kernel on the unpadded slab box, as Quantum ESPRESSO's exact exchange
  (its `assume_isolated='2D'` cuts the Hartree, local and Ewald terms only),
  for comparisons.
- **The loop.** At the Gamma point without k-point input and without spin the
  orbitals are real and the cluster loop runs with the FFT pair solver;
  otherwise (k-points, or spin at the Gamma point as the 1x1x1 mesh) the
  orbitals are complex Bloch factors and the inner loops are the k-point SCF
  with `a_x V_ACE` of every k-point (`M = -U^H W = L L^H`, `xi = W L^-H`) in
  its Bloch Hamiltonian. The stop rule, the adaptive inner criterion and the
  outer DIIS (one block per k-point) are those of the clusters, with
  k-averaged energies.
- **Output.** After the loop `parsec.out` gives the valence band maximum,
  the conduction band minimum (an occupation above one half counts as
  occupied) and the gap, direct or indirect with the smallest direct one.
- **Forces.** Hellmann-Feynman forces of the hybrid orbitals and density; the
  exchange operator of norm-conserving pseudopotentials adds no explicit term
  (see [FORCES_AND_RELAXATION.md](FORCES_AND_RELAXATION.md)). Forces,
  relaxation and dynamics run at the Gamma point (also with spin) and with
  k-points (spin-unpolarized; with spin on the Fortran-derived routines, as
  for semilocal functionals). No stress.
- `Restart_Run` starts the semilocal start loop from the stored potential, as
  for clusters. `occup.in` (a negative `Fermi_Temp`) is taken at the Gamma
  point without k-point input and refused with k-points.
- **Against Quantum ESPRESSO 7.5** with the same FHI98PP pseudopotentials:
  bulk Si, C, MgO and an 8-atom Si cube with Gamma-centred meshes 1x1x1 to
  4x4x4, the h-BN monolayer at 3x3x1 and 6x6x1, triplet O2 in a 12-bohr
  cube at the Gamma point; PBE0, HSE06, HF and B3LYP with each of QE's
  treatments of the singularity, `Exx_Auxiliary_Exponent = 10/ecutwfc`, the
  slab with `Exx_Slab_Coulomb: periodic`. `E_hyb - E_PBE` agrees within
  `4e-5 Ry` for Si, C and Si8 (PBE0 within `8e-6`) and `2.6e-5` for MgO,
  `a_x E_x` within `6e-5 Ry` (HF), the gaps within 0.7 meV (1.9 meV MgO).
  HSE06 agrees as well once QE's own artefact is removed: QE skips its
  short-range PBE exchange at grid points where the density gradient
  vanishes (atom sites, bond centres), which lowers its HSE energies by up
  to `3.2e-4 Ry` (O2) unless the atoms are moved off its grid points.
  h-BN: within `1.3e-4 Ry` (PBE0, HSE06) against QE at 100 Ry, whose own
  cutoff error is `9.7e-5`; against QE at 160 Ry PBE0 `+2.9e-5` (3x3x1) and
  `+7e-6` (6x6x1), HSE06 `+6e-6` (6x6x1); HF `2.8e-4`; the bands at K within
  1 meV. The
  hybrid-minus-PBE force agrees within `4e-6 Ry/bohr` (Si with a displaced
  atom, 2x2x2 and 3x3x3), and finite differences of the energy give the
  analytic PBE0 force to `1.3e-6 Ry/bohr`.

### What is refused

Each of these is refused by name when the input is read:

- wires (`Boundary_Conditions: wire`); on a periodic cell `Exx_Pair_Solver: cg`
  (its pair potentials are solved by FFT), a `Kpoint_List` that is not a full
  uniform mesh written out with equal weights, and `--mpi`; `Exx_Divergence`
  and `Exx_Auxiliary_Exponent` on an isolated system, `Exx_Slab_Coulomb`
  outside a slab;
- spin-orbit coupling (`SO_PSP`, `SO_from_scratch`) and non-collinear
  magnetism: the operator acts on real scalar orbitals;
- FDET (`Potential_Field`, and so AFM scans) and LDA+U (`ldaplusu_u`,
  `ldaplusu_j`): their terms are not combined with the exchange operator;
- `Energy_Refinement`, in the input or as `--energy-refinement`: the hybrid
  is already self-consistent;
- `Van_Der_Waals`: PARSEC's Tkatchenko-Scheffler correction is defined for
  PBE;
- `Polarizability`, `nscf_kpoints`, `Output_GW` and band structures: not
  implemented together with the Fock loop.

### Pseudopotentials

There are no hybrid norm-conserving pseudopotentials in the libraries of this
repository; every hybrid reads PBE files (correlation label `pb` or `pbe`),
as Quantum ESPRESSO's hybrid runs usually do, and B3PW91 also PW91 files
(`pw`, `pw91`, `pwgga`). A CA (LDA) file is used with a WARNING that names it,
as in a PBE run; a file of any other functional is refused when the input
is read. FHIPP and old-format Martins files carry no label.

### Diagnostics of the Fock loop

After the loop `parsec.out` may add:

- a NOTE when dexx rose more than tenfold above its smallest earlier value:
  the state the loop approached was unstable under exact exchange, and the
  loop went on to a lower one;
- a WARNING when the loop stopped unconverged while its energy steps shrink
  geometrically: the ratio per outer iteration and the energy still to be
  gained; raise `Exx_Max_Iter`;
- a WARNING for fractional final occupations.

Exact exchange penalizes fractional occupations. A symmetric ensemble with
a half-filled degenerate level, which the PBE start can give at finite
`Fermi_Temp`, can therefore be a stationary point but a saddle of the hybrid
functional, with a symmetry-broken state of lower energy. A ring of 12
hydrogen atoms (bond 1.40 bohr, PBE0, 80 K) shows all three cases: dexx
first falls to about `3e-7 Ry`, then rises to `3e-2 Ry` as the symmetry
breaks, and the bond-alternating state that follows lies about 65 mRy lower;
its soft rotation then shrinks dexx by only 0.90 per outer iteration, so the
loop needs about 50 outer iterations, as Quantum ESPRESSO, which needs 66.
With the default `Exx_Tolerance` such a loop can stop at the symmetric
saddle; the fractional-occupation warning says so. The symmetry sectors of
`main.py` (`--symmetry auto`) keep the reflection symmetry of the atoms in
every orbital, so they cannot reach a state that breaks it; `--symmetry off`
can.

### Comparison with Quantum ESPRESSO

For H2, H2O, CH4, CO, N2, O2 (triplet) and benzene with PBE0, HSE06, HSE03,
B3LYP, B3LYP5, B3PW91 and HF, against Quantum ESPRESSO 7.5 with the same
FHI98PP pseudopotentials and geometries, `E_hyb - E_PBE` agrees within
`9.4e-5 Ry` in all 49 cases (the largest, B3LYP and B3LYP5 of O2, are QE's
cutoff error for LYP: `1.1e-5 Ry` against QE at 250 Ry), the Fock energies
`a_x E_x` within `5.9e-5 Ry` and the HOMOs within 1.94 meV (HSE03 against
a reference derived from QE's HSE and libxc's HSE03, since QE has no HSE03
of this form). Inputs, references and tables: [`examples/0d_hybrid_functionals`](../../examples/0d_hybrid_functionals/).

## Functionals of libxc

```text
LibXC_Functional: gga_x_pbe gga_c_pbe
```

runs the SCF with functionals of [libxc](https://libxc.gitlab.io) (S. Lehtola,
C. Steigemann, M. J. T. Oliveira, M. A. L. Marques, SoftwareX 7, 1 (2018);
MPL-2.0), an optional library that the code loads when such a functional is
asked for ([V_xc/libxc_library.py](V_xc/libxc_library.py), ctypes;
[V_xc/libxc.py](V_xc/libxc.py)). Every functional of libxc of the LDA and GGA
families runs where the functionals of this code run (clusters and
periodic cells, unpolarized and collinear spin, forces, relaxation and
dynamics, both CLIs), except spin-orbit coupling, non-collinear magnetism,
FDET, the TS van der Waals correction (PBE of this code only, as 1.5) and
`--mpi` and the MPI runner of several nodes (`scripts/run_multi_gpu_scf.sh`, hybrids too: not tested
there), which are refused with a libxc functional; a libxc hybrid runs through the Fock loop of
[Self-consistent hybrid functionals](#self-consistent-hybrid-functionals) with
its exact exchange; the meta-GGAs run through the meta-GGA Hamiltonian
(`V_xc/meta_gga.py`, `SCF/meta_gga.py`: `tau` of the orbitals, `-div(vtau grad)`
in every Hamiltonian; clusters and periodic cells, see
[Meta-GGA functionals](#meta-gga-functionals)), the meta-GGA hybrids (TPSSh,
r2SCAN0, SCAN0, M06-2X, ...) of clusters through the Fock loop with
meta-GGA inner loops. In GPU runs (`main.py`, CuPy)
libxc evaluates on the GPU when the library is a CUDA build (libxc >= 7.1).

### Keywords

| keyword | default | meaning |
|---|---|---|
| `LibXC_Functional` | - | one or more libxc functionals, by name (`GGA_X_PBE`, `gga_x_pbe`, `XC_GGA_X_PBE`) or number (`101`), separated by blanks, commas or `+`; their energies are added (an exchange and a correlation functional, or one exchange-correlation functional such as `hyb_gga_xc_b3lyp`) |
| `Use_LibXC`, `Func_Id_X`, `Func_Id_C` | `.false.`, -1, -1 | PARSEC 1.5's keywords: `Use_LibXC: .true.` with the libxc numbers of the exchange and the correlation functional; -1 takes 1.5's defaults (1 = LDA_X; 9 = LDA_C_PZ, 12 = LDA_C_PW with spin), 0 leaves the slot empty. With `.false.` the ids are ignored, as in 1.5. `LibXC_Functional` and `Func_Id_*` together are refused |
| `LibXC_Enforce_FHC` | by functional | the Fermi-hole-curvature condition of meta-GGAs, `tau_s >= tau_W,s = sigma_ss/(8 n_s)`: `.true.` floors `tau` of each channel at `tau_W` with the exact derivatives (the meta-GGA grid layer); `.false.` evaluates the functional at the `tau` of the orbitals, as Quantum ESPRESSO (which switches off libxc 7's default clamp) and the native SCAN do. Without the keyword: the floor for a functional whose derivatives jump at `tau = tau_W` (libxc clamps `z = tau_W/tau` at 1 inside TPSS-type correlation: TPSS, revTPSS, TPSSh, rSCAN; a probe of the functional finds them), so that all its components see `z <= 1`; none for SCAN, r2SCAN, M06-L, MN15-L, M06-2X and the other smooth ones. libxc's own clamp (`sigma -> 8 n tau`, not carried into the derivatives; NaN in the polarized SCAN and r2SCAN correlation where it leaves a zero total gradient) is not used. No effect on LDA and GGA |
| `LibXC_Tau_Threshold` | 1e-12 | the absolute floor of the `tau` of each channel of a meta-GGA (Hartree/bohr^3; `tau_minimum` of the meta-GGA grid layer): a point below it keeps the energy of `tau = t`, with `vtau = 0` there, the exact derivative of that energy; below it the binding floors `tau` at 1e-20 (libxc's default threshold) on the host and the device alike. Where every occupied orbital has a zero gradient at a density maximum (the bond centre of Si at Gamma) `tau` and `sigma` are round-off; at libxc's floor `z = tau_W/tau` of TPSS is a ratio of round-off, `vtau` and `vsigma` reach 1e9 to 1e13 and the SCF does not converge, while at `t` `z` goes to 0 with `sigma`. Quantum ESPRESSO drops a point whose `tau` is at most 1e-12: a plane-wave grid point weighs nothing there, a real-space one `h^3` (0.03 Ry of Si at Gamma at h 0.4), so it keeps its energy here |
| `LibXC_Density_Threshold` | libxc's, per functional | libxc's density threshold of every component (`xc_func_set_dens_threshold`), e/bohr^3: below it a point contributes nothing, and libxc floors each spin density at it |

`Correlation_Type` beside a libxc functional does not select the functional
(a warning says so), as in 1.5. With `Use_LibXC` and an exchange-correlation
functional in `Func_Id_X` (`402`, `457`), 1.5's default `Func_Id_C` (LDA_C_PZ,
LDA_C_PW with spin) is added to it, as 1.5 does, with a warning: give
`Func_Id_C: 0` (or `LibXC_Functional`) for the functional alone. In the settings of a run the functional is
`SCFSettings.xc_functional = "libxc:gga_x_pbe+gga_c_pbe"`, with
`;fhc=on|off`, `;threshold=<t>` and `;tau=<t>` where those keywords are given;
`Generate_Complete_Parsec_In` writes `LibXC_Functional` back.
`python -m parsec_python.V_xc.libxc [--list] [names]` prints the library the
code finds, whether it evaluates on the device, and the functionals with
their families, CAM coefficients and references; each run prints the
components, their references and libxc's citation under "Correlation data".

### What the code does with them

- **Variables and units.** libxc returns the energy per particle `zk` and the
  derivatives of the energy per volume in Hartree, in its spin layout
  (`rho` and `tau` interleaved per point, `sigma = (uu, ud, dd)`); the
  evaluator (`LibxcSemilocal`, the `SemilocalXC` interface of
  [V_xc/semilocal.py](V_xc/semilocal.py)) returns `e = zk (n_u + n_d)` and the
  derivatives per channel, and the grid functions of every semilocal
  functional of this code take them over: `sigma = |D n|^2` with the
  centered stencil of `Expansion_Order`, the potential
  `v = vrho + D^T (2 vsigma D n)` (spin: `v_a = vrho_a + D^T (2 v_aa D n_a + v_ab D n_b)`),
  the exact derivative of the discrete energy, in Rydberg, the frozen core
  density added (half per channel).
- **First iteration.** LDA and GGA functionals are used from the first
  iteration. PARSEC 1.5 evaluates the first XC of every libxc run with PBE
  (101 + 130, `potential/hartree.F90`), which a density functional does not
  need; a meta-GGA, whose `tau` needs orbitals, starts with PBE as there.
- **Hybrids.** A libxc hybrid (`hyb_` names) is a
  [`HybridFunctional`](V_xc/hybrid.py) with libxc's CAM coefficients
  (`xc_hyb_cam_coef`: `alpha` at long range, `alpha + beta` at short range,
  `omega` in bohr^-1) and libxc's functional as its semilocal part (libxc's
  energy of a hybrid is its semilocal part). The Fock loop, its keywords
  (`Exx_*`), its refusals and its GPU path are those of the hybrids of this
  code; it starts from a PBE SCF. Periodic cells (bulk and slabs, at the Gamma point and with k-points) run the
  hybrid GGAs of libxc through the periodic Fock loop: `hyb_gga_xc_pbeh` equals the code's PBE0 to round-off with
  libxc's PBE digits in the code's PBE (H2 in a 6-bohr box, Gamma and a 2x1x1 mesh,
  `tests/unit/test_libxc.py`) and lies 5e-8 Ry below it without them (Si, two atoms, Gamma and 2x2x2, one A100), the
  offset of the two PBEs. PARSEC 1.5 evaluates libxc's hybrid GGAs as
  plain GGAs without their exact exchange. A meta-GGA hybrid (`hyb_mgga_`)
  runs the same Fock loop with the meta-GGA loops as inner loops: every
  Hamiltonian holds `a_x V_ACE` and `-div(vtau grad)`, the energies remove the
  band energy of both, the mixed `vtau` goes from one inner loop to the next,
  and the first inner loop starts as every meta-GGA loop starts (the PBE
  potential of the start density, no orbital term yet); its inner loops
  restart their mixer (`Exx_Mixer_History: carry` is refused for them). The
  density gradient follows the meta-GGA's form (`auto`: the orbital form for
  r2SCAN0, SCAN0, M06-2X, the density form for TPSSh); in the orbital form the
  inner loops converge as those of a GGA hybrid (NH3 r2SCAN0, h 0.16, one
  A100: 4 outer iterations of 12-20 inner ones, 88 s, against 3062 s in the
  density form). An exchange-only hybrid
  (`hyb_gga_x_s12h`, `hyb_mgga_x_scan0`) beside a correlation functional
  takes the exact exchange of that one component once: libxc's value of the
  component already holds its reduced semilocal exchange.
  `Exact_Exchange_Fraction` and `Screening_Parameter` are refused with a
  libxc hybrid (libxc's own variants, such as `hyb_gga_xc_pbeh` or
  `hyb_gga_xc_hse12`, carry other coefficients).
- **GPU.** `acceleration/V_xc/device_libxc.py` evaluates libxc's CUDA build
  on the device pointers of the CuPy arrays, on the full grid or once per
  symmetry orbit in the sectors, with the stencils of the code's device
  functionals; a library without device support makes the host evaluate
  libxc (the output names the implementation), and `PARSEC_CUPY_XC=off` keeps
  every functional on the host.
- **Pseudopotentials.** A libxc LDA reads CA/LDA files, every other libxc
  functional PBE files; a file of another label is used with a warning, as
  `pseudo.f90` warns.

### What is refused

Of the 712 functionals of libxc 7.1.2, 589 run: 61 LDA, 205 GGA and 162
meta-GGA functionals, and 161 hybrids (4 LDA, 105 GGA, 52 meta-GGA; exact
exchange with the kernels `1/r`, `erf`, `erfc`, `1/r + erf` or `1/r + erfc`);
123 are refused: 70 kinetic-energy functionals, 19 of one or two dimensions,
13 with VV10, 9 fitted with an empirical dispersion term, 6 Yukawa-screened
hybrids and 6 without an energy or a potential.

When the input is read: a name or number libxc does not have; kinetic-energy
functionals; functionals with the nonlocal VV10 correlation (wB97M-V,
B97M-V, ...); functionals fitted together with an empirical dispersion term
that libxc does not hold (wB97X-D, wB97X-D3, B97-D, SSB-D, oTPSS-D, ...: the
names ending in `_d`, `_d3`); hybrids with a Yukawa-screened kernel
(CAMY/LCY); functionals of one or two dimensions; functionals without an
energy or a potential in libxc (`gga_x_lb`); two hybrid components in one
functional; meta-GGA hybrids of periodic cells; the TS van der Waals
correction (`Van_Der_Waals`, PBE of this code only, PARSEC 1.5's error 158);
spin-orbit coupling, non-collinear magnetism, FDET and `--mpi` together with
a libxc functional (not tested); without the library, every libxc keyword,
with how to install it.

### Meta-GGAs and meta-GGA hybrids against Quantum ESPRESSO

The molecules of the meta-GGA references (FHI98PP `NCPPs/PBE` H, C, N, O;
`Grid_Spacing` 0.16 bohr, `Expansion_Order` 12, sphere 11.14-11.53 bohr;
Quantum ESPRESSO 7.5 + libxc at 150 Ry, ecutfock 300 Ry, 20 bohr box, its
hybrids TPSSh and r2SCAN0 stock, SCAN0 from a pw.x that takes libxc's
exchange-only hybrid once); `E - E_PBE` in mRy, each code against its own
PBE; the default form of the density gradient (`auto`: density form for TPSS
and TPSSh, orbital form for the others); one A100, `Exx_Tolerance` 1e-8 Ry,
`Convergence_Criterion` 1e-6 Ry, `Max_Iter` 200. M06-2X and the O2 TPSSh and
r2SCAN0 have no Quantum ESPRESSO reference (Quantum ESPRESSO 7.5 aborts on
M06-2X, and its O2 TPSSh and r2SCAN0 diverge before the exact exchange starts):

| molecule | functional | dE this code (mRy) | dE QE (mRy) | ours - QE (mRy) | a_x E_x - QE (mRy) | inner iterations | s |
|---|---|---|---|---|---|---|---|
| H2O | tpssh | -8.113 | -8.186 | +0.073 | +0.005 | [6, 9, 14] | 39.1 |
| H2O | r2scan0 | +11.265 | +11.303 | -0.038 | +0.014 | [11, 12, 17, 8] | 53.7 |
| H2O | scan0 | +14.166 | +14.242 | -0.076 | +0.174 | [6, 13, 12, 6] | 52.2 |
| H2O | m062x | -14.492 | - | - | - | [14, 10, 12, 11, 9, 8] | 58.5 |
| H2O | r2scan | -2.132 | -2.038 | -0.094 |  |  | 32.1 |
| H2O | scan | +1.397 | +1.543 | -0.147 |  |  | 37.0 |
| H2O | tpss | -11.075 | -11.145 | +0.070 |  |  | 30.5 |
| CH4 | tpssh | -41.476 | -41.426 | -0.050 | -0.003 | [12, 10, 15, 9] | 41.5 |
| CH4 | r2scan0 | -3.799 | -3.745 | -0.054 | +0.001 | [7, 14, 9, 7] | 53.4 |
| CH4 | scan0 | -5.614 | -5.488 | -0.126 | +0.019 | [7, 21, 14, 8] | 58.1 |
| CH4 | m062x | +3.441 | - | - | - | [42, 53, 39, 29, 24, 9] | 124.0 |
| CH4 | r2scan | -1.511 | -1.422 | -0.088 |  |  | 34.5 |
| CH4 | scan (not converged) | -4.046 | -3.821 | -0.225 |  |  | 112.7 |
| CH4 | tpss | -39.429 | -39.357 | -0.072 |  |  | 29.9 |
| N2 | tpssh | +18.447 | +18.419 | +0.028 | +0.003 | [6, 6, 9] | 26.5 |
| N2 | r2scan0 | +61.904 | +61.921 | -0.018 | -0.010 | [8, 6, 17, 5] | 34.4 |
| N2 | scan0 | +64.839 | +65.073 | -0.234 | -0.162 | [5, 10, 16, 11] | 32.0 |
| N2 | m062x | +19.136 | - | - | - | [5, 8, 11, 11, 10, 4] | 36.8 |
| N2 | r2scan | +35.224 | +35.240 | -0.016 |  |  | 27.6 |
| N2 | scan | +38.779 | +39.273 | -0.493 |  |  | 34.9 |
| N2 | tpss | +8.420 | +8.392 | +0.028 |  |  | 25.0 |
| CO | tpssh | +9.782 | +9.795 | -0.013 | +0.005 | [6, 9, 9] | 34.0 |
| CO | r2scan0 | +43.956 | +44.056 | -0.100 | -0.005 | [6, 15, 13, 10, 9] | 46.4 |
| CO | scan0 | +46.601 | +46.508 | +0.093 | -0.107 | [7, 17, 120, 168, 13] | 137.2 |
| CO | m062x | +3.032 | - | - | - | [6, 13, 13, 13, 11, 9] | 51.6 |
| CO | r2scan | +19.591 | +19.736 | -0.145 |  |  | 32.4 |
| CO | scan (not converged) | +22.765 | +22.822 | -0.057 |  |  | 94.8 |
| CO | tpss | +1.575 | +1.589 | -0.014 |  |  | 28.3 |
| NH3 | tpssh | -16.874 | -16.873 | -0.001 | +0.003 | [7, 15, 11] | 50.0 |
| NH3 | r2scan0 | +11.177 | +11.158 | +0.019 | +0.007 | [12, 20, 16, 16] | 82.8 |
| NH3 | scan0 | +12.380 | +12.512 | -0.132 | -0.057 | [10, 15, 26, 12] | 85.7 |
| NH3 | m062x | +3.175 | - | - | - | [7, 11, 13, 14, 11, 6] | 76.7 |
| NH3 | r2scan | +4.850 | +4.828 | +0.022 |  |  | 64.9 |
| NH3 | scan (not converged) | +6.216 | +6.491 | -0.275 |  |  | 148.1 |
| NH3 | tpss | -17.648 | -17.640 | -0.008 |  |  | 39.7 |
| O2 | tpssh | +17.023 | - | - | - | [7, 12, 10] | 41.9 |
| O2 | r2scan0 | +55.269 | - | - | - | [7, 9, 11, 9, 7] | 52.6 |
| O2 | scan0 | +59.848 | +58.758 | +1.090 | +0.066 | [6, 16, 13, 11, 9] | 63.3 |
| O2 | m062x | +31.271 | - | - | - | [6, 12, 11, 8, 10, 5] | 58.6 |
| O2 | r2scan | +13.137 | +13.351 | -0.214 |  |  | 36.5 |
| O2 | scan | +19.193 | +18.074 | +1.119 |  |  | 43.4 |
| O2 | tpss | +1.759 | +1.681 | +0.079 |  |  | 33.1 |

r2SCAN, TPSS and their hybrids agree within 0.1 mRy (r2SCAN O2 0.2 mRy) and
the exact exchange within 0.02 mRy; SCAN and SCAN0 within 0.1-0.5 mRy and O2
1.1 mRy, the grid sensitivity of SCAN (Quantum ESPRESSO's own SCAN
`E - E_PBE` of O2 moves by 0.7 mRy between 150 and 300 Ry; see
[Accuracy and grid spacing](#accuracy-and-grid-spacing)). The SCAN runs of
CH4, CO and NH3 stop at `Max_Iter` 200 with the energy settled (SCAN's
residual floor, see [Method](#method)). The inputs and logs of the
semilocal rows, TPSSh and r2SCAN0 (with the h = 0.13 bohr rows, forces and
timings) are [`examples/0d_meta_gga`](../../examples/0d_meta_gga/README.md).

### libxc's versions of this code's functionals

| this code | libxc | difference |
|---|---|---|
| `ca` | `lda_x lda_c_pz` | none: round-off |
| `pw92` (`pl`) | `lda_x lda_c_pw_mod` | Fortran PARSEC's digits of a few PW92 constants (6th to 9th digit) |
| `pbe` | `gga_x_pbe gga_c_pbe` | the PW92 `A` of the correlation, `(1 - ln 2)/pi^2 = 0.0310906908` here, `0.0310907` in libxc: 3e-7 of the local correlation (H2 at h = 0.5 bohr: 1e-7 Ry) |
| `pw91` | `gga_x_pw91 gga_c_pw91` | Fortran PARSEC's digits of a few constants |
| `b3lyp`, `b3lyp5`, `cam-b3lyp` | `hyb_gga_xc_b3lyp`, `hyb_gga_xc_b3lyp5`, `hyb_gga_xc_cam_b3lyp` | none: round-off |
| `pbe0` | `hyb_gga_xc_pbeh` | as PBE |
| `hse06` | `hyb_gga_xc_hse06` | libxc's wPBEh hole and rescaling of s, see `HybridFunctional.variants` |

With libxc's digits put into the code's functionals the two agree to
round-off: pointwise to 2e-12 relative, as grid fields to 1e-12, and the SCF
of H2 (PBE, the B3LYP Fock loop) and of an H atom in a periodic box to 1e-9 Ry
([tests/unit/test_libxc.py](tests/unit/test_libxc.py)). A fully polarized
density differs by libxc's floor of the empty channel at its density
threshold: the spin PBE H atom by 2.6e-9 Ry at libxc's threshold and 5e-13 Ry
at fixed density with `LibXC_Density_Threshold: 1e-30`. On the molecules of
[`examples/0d_hybrid_functionals`](../../examples/0d_hybrid_functionals/)
(GPU, h = 0.16 bohr) libxc's B3LYP gives the code's B3LYP to the printed
digits (O2 triplet included), libxc's PBE and PBEh the code's PBE and PBE0
within 1e-8 to 1.3e-7 Ry (the `A` above), and libxc's HSE06 and HSE03 through
this code's exact exchange reproduce Quantum ESPRESSO with libxc within
2.1e-5 Ry in `E_hyb - E_PBE` and 1.3e-5 Ry in `a_x E_x`.

libxc 7.1 (7.1.2 checked) evaluates its polarized LYP inaccurately where one
spin channel is almost empty (`n_u = 4.2e-11` beside `n_d = 9.1e-5`: on the
host `vrho_u` 3e-4 and `vsigma_uu` 33 % off a 60-digit evaluation, on the
device `vsigma_uu` 33 % off as well; the host error falls as `n_u^(-11/3)`,
2.5e-11 at `n_u/n = 5e-5`); libxc 6.2.2 and the code's own LYP give the exact
values. Host and device therefore differ there (the B3LYP spin potential of
a test field by 8.5e-6 of its largest value). The points carry no density to
speak of: the spin B3LYP H atom differs by less than 1e-10 Ry between host
and device, O2 (triplet) libxc B3LYP equals the code's B3LYP to the printed
digits, and the GPU tests compare spin fields whose channels both hold a
share of the density.

### Installation

libxc >= 7.0 for the host, a CUDA build of libxc >= 7.1 for the GPU:

```bash
bash scripts/build_libxc.sh --prefix $HOME/libxc-7.1.2-cpu             # host build (CMake)
bash scripts/build_libxc.sh --cuda --cuda-arch 80 --prefix $HOME/libxc-7.1.2-cuda-sm80   # GPU (A100)
export PARSEC_LIBXC_LIBRARY=$HOME/libxc-7.1.2-cuda-sm80/lib/libxc.so
python -m parsec_python.V_xc.libxc
```

The script downloads the source from libxc's GitLab and builds energy and
potential only (`MAXORDER 1`); one CUDA build evaluates host and device
arrays. The CUDA build is compiled without fused multiply-adds
(`--fmad off`, as the device kernels of this code; `--fmad on` for nvcc's
default), so that host and device round alike. Without `PARSEC_LIBXC_LIBRARY` the code looks for `libxc.so`
(`xc.dll`, `libxc.dylib`) on the loader's path. conda-forge's `libxc-c`
(7.1.2 for Linux with cpu and cuda variants, and for Windows, cpu) works as
well: `conda install -c conda-forge libxc-c` and point the variable at its
`lib/libxc.so` (Windows: `Library\bin\xc.dll`). On Perlmutter the builds of
the script are `<prefix>/libxc/7.1.2-cuda-sm80/lib/libxc.so`
(CUDA 12.9, sm_80; it links `libcudart.so.12`, the runtime CuPy 14 loads on the
A100 nodes, so one runtime serves both) and `.../7.1.2-cpu/lib/libxc.so` (host
only: `libm` and `libc`, it loads on CPU nodes). The script compiles with the
plain `gcc` (and `g++` as the host compiler of `nvcc`): Cray's `cc` wrapper
links the CUDA runtime of the loaded `craype-accel` module into any library,
and checks with `ldd` that a host build needs no CUDA runtime.

## Meta-GGA functionals

```text
Correlation_Type: scan
```

or a meta-GGA of libxc (`LibXC_Functional: mgga_x_r2scan mgga_c_r2scan`, see
[Functionals of libxc](#functionals-of-libxc)) runs the SCF with a functional
of the kinetic-energy density of the occupied orbitals,
`tau = (1/2) sum_n f_n |grad psi_n|^2` (Hartree, libxc's convention), and of
the Laplacian of the density where the functional asks for it. Isolated
systems and periodic cells (see [Periodic cells](#periodic-cells)),
unpolarized and collinear spin, forces (with a nonlinear core
correction), relaxation and dynamics, on the CPU and on the GPUs of one node,
on the full grid and in the symmetry sectors; meta-GGA hybrids of clusters
with the functionals of libxc ([Functionals of libxc](#functionals-of-libxc))
([V_xc/meta_gga.py](V_xc/meta_gga.py), [SCF/meta_gga.py](SCF/meta_gga.py),
[acceleration/Hamiltonian/meta_gga_device.py](acceleration/Hamiltonian/meta_gga_device.py)).

### Functionals

| `Correlation_Type` | functional | evaluation |
|---|---|---|
| `scan` | SCAN of Sun, Ruzsinszky and Perdew (PRL 115, 036402 (2015)) | native ([V_xc/scan.py](V_xc/scan.py)): libxc's formulas (`mgga_x_scan`, `mgga_c_scan`) to 5e-10 relative wherever `n > 1e-8`, derivatives by complex steps; on the host and the GPU |
| a libxc `mgga_` functional | e.g. r2SCAN, TPSS, M06-L | libxc, on the GPU with a CUDA build |

The Fermi-hole-curvature condition `tau >= tau_W` is not enforced for the
native SCAN and the libxc functionals smooth across `tau = tau_W` (as Quantum
ESPRESSO); a libxc functional with a kink there (TPSS-type correlation, which
clamps `z = tau_W/tau` at 1) has `tau` of each channel floored at `tau_W`
with the exact derivatives, so that all its components see `z <= 1`
(`LibXC_Enforce_FHC` sets it either way). `tau` of each channel of a libxc
meta-GGA is also floored at `LibXC_Tau_Threshold` (1e-12 Hartree/bohr^3): where
every occupied orbital has a zero gradient (the bond centre of Si at the
Gamma point) `tau` and `sigma` are round-off, and `z` of TPSS would be a
ratio of round-off; the point keeps the energy of the floored input, with
`vtau = 0` there.
Pseudopotentials: PBE files (a meta-GGA reads the PBE label, as for the
hybrids).

### Method

- **Discretization.** `tau` from the centered first-derivative stencil `G`
  of `Expansion_Order` applied to the orbitals; the energy
  `E = h^3 sum_i e(n_i, sigma_i, tau_i)`; the orbital term of the Hamiltonian
  `-div(vtau grad)` as `sum_c G_c^T diag(vtau) G_c` (Rydberg), Hermitian and
  the exact derivative of the discrete energy with respect to the orbitals.
  The frozen core density enters `n` and the density gradient, never `tau`.
- **The density gradient.** Two forms (`PARSEC_META_GGA_GRADIENT`):
  `orbital` takes `g = (d/h^3) sum_n f_n 2 q_n (G q_n)` from the orbitals,
  `density` takes the stencil gradient of the density, `G n`, as the GGAs do;
  `auto` (the default) takes the orbital form for a functional whose
  derivatives are continuous across `tau = tau_W` and the density form for
  one with a kink there (TPSS-type correlation clamps `z = tau_W/tau <= 1`;
  `LibXC_Enforce_FHC: .true.` floors `tau` at `tau_W`), decided by a probe of
  the functional's derivatives at `tau_W (1 -+ 1e-6)` on a few densities and
  gradients; `parsec.out` names the form and why. In the orbital form every
  point of a one-orbital region sits at `tau = tau_W` up to round-off, and a
  kink there makes those points change sides from one iteration to the next
  (libxc TPSS on H2: the residual stays at 1e-2).
  On the grid only the orbital form keeps `tau >= |g|^2/(8n)` (Cauchy-Schwarz;
  equality where one orbital carries the density); with `G n`, `tau - tau_W`
  of the grid is noise of either sign in the tails and near the nuclei, which
  `alpha = (tau - tau_W)/tau_unif` of SCAN amplifies where the density is
  small, and the SCF gets worse as the grid gets finer (H2O on an A100, SCAN:
  SRE 1.7e-5 after 80 iterations at h = 0.20 bohr, 1.7e-4 after 80 at 0.16
  with a passing excursion to +2782 Ry, 1e-3 after 120 at 0.13; libxc r2SCAN
  diverges at 0.11), while the orbital form converges as a GGA does (H2O
  h = 0.30: 1e-7 in 24 iterations; O2 h = 0.40, spin: 1e-7 in 25; CH4
  h = 0.40: 1e-7 in 28; r2SCAN H2O h = 0.30: 1e-7 in 19). The orbital form's
  potential is `vrho` alone, and the Hamiltonian holds the first-order term
  `2 sum_c [W_c G_c + G_c^T W_c]` of `W = de/dg`. The two forms differ by the
  grid error (first-order energy differences of at most 0.04 mRy at
  h = 0.20, 0.006 at 0.16 and 0.001 at 0.13 on the molecules of the
  references). The functionals with a kink at `tau_W` keep the density form
  and its convergence (libxc TPSS, H2O h = 0.30: 1e-7 in 19 iterations).
  SCAN itself is sensitive to `alpha` near 1 (its switching function): on CH4
  the energy converges to 1e-10 Ry within 16-30 iterations, while the SRE
  stays at 5e-6 to 2e-5 (h = 0.16-0.25), set by points of density 1e-5 to
  1e-3; r2SCAN, built to remove that sensitivity, converges to 1e-8 in 23
  iterations with the same SCF.
- **First iteration.** PBE of the start density without orbital terms (as
  PARSEC 1.5 starts its libxc meta-GGAs); after an ionic move the mixed
  `vtau` (and `W`) of the structure before.
- **Mixing.** `Mixing_Method`'s mixer sees `(V, vtau[, W])` of all channels
  joined, one history; Anderson's least squares weigh every value by the
  density (`PARSEC_META_GGA_ANDERSON=weighted`, the default; `plain` for
  PARSEC's metric). The stop test is PARSEC's on `V`; the residual of `vtau`
  is recorded. The weights let the low-density values of `vtau` and `W`
  drift in the Anderson combination; on fine grids that can produce states
  localized on a few points (SCAN H2O at h = 0.13 bohr, orbital form: two
  passing excursions and SRE 4.6e-5 after 80 iterations, where `plain`
  converges in 54). With the orbital form neither choice wins everywhere on
  an A100 (iterations to 1e-6 with `weighted` / `plain`: SCAN H2O h = 0.20
  37 / 27, h = 0.16 45 / 61, O2 h = 0.16 41 / 45), so `weighted` stays the
  default; on fine grids use `plain` or r2SCAN (r2SCAN H2O h = 0.13 with
  `plain`: 53 iterations).
- **Energy.** The band energy holds `<psi|H_tau|psi>`; the total removes
  `2 h^3 sum (vtau_in tau + W_in . g)` (printed as `Integral_{2 vtau*tau}`
  under the energy corrections of `parsec.out`).
- **Eigensolver.** The filter's upper bound takes in Gershgorin's bound of
  the orbital term ([acceleration/README.md](acceleration/README.md#meta-gga-functionals-on-the-device)).

### Accuracy and grid spacing

A meta-GGA needs a finer grid than a GGA for the same accuracy of energy
differences. Its energy is the sum `h^3 sum_i e(n_i, sigma_i, tau_i)` over
the grid points; `tau` and the iso-orbital indicator
`alpha = (tau - tau_W)/tau_unif` vary on shorter length scales than the
density, and the sum carries a quadrature (aliasing) error that changes sign
from one grid spacing to the next (a plane-wave code has the same error on its
dense FFT grid). Against Quantum ESPRESSO 7.5 with libxc and the same
FHI98PP potentials (`NCPPs/PBE` H, C, N, O; 300 Ry, 20 bohr box; H2O, CH4,
NH3 and the O2 triplet), the difference of `E_mGGA - E_PBE` (mRy) and of the
HOMO (meV):

| functional | h = 0.20 bohr | 0.16 | 0.13 |
|---|---|---|---|
| r2SCAN | 0.08-0.18, O2 1.0 / HOMO 0.3-2, O2 14 | 0.03-0.11, O2 0.34 / 0.1-1.4 | 0.01-0.05, O2 0.11 / 0.1-2.7 |
| TPSS (H2O; O2 at 0.16) | 0.18 / 0.3 | 0.01, O2 0.03 / 0.3-0.9 | 0.02 / 0.9 |
| SCAN | 0.2-0.8 / 5-50 | 0.1-0.9 / 1-40 | 0.03-0.2, O2 0.2 / 0.3-25 |

The derivative stencil of `sigma` and `tau` (against exact derivatives of the
same orbitals: <= 0.07 mRy, < 0.3 meV), the boundary of the domain (3 bohr
more radius: < 1e-6 Ry, 0.05 meV) and the form of the density gradient (<=
0.04 mRy at 0.20, 0.006 at 0.16) are not what is left: with the energy
density summed on a twice finer grid (the orbitals interpolated spectrally),
r2SCAN and TPSS agree with Quantum ESPRESSO to 0.04 mRy and 2.5 meV at every
spacing above (O2 at h = 0.20: 1.045 -> -0.027 mRy). SCAN's steep switching
function in `alpha` makes it grid-sensitive in every code (Quantum ESPRESSO's
own SCAN `E - E_PBE` moves by 0.1-0.5 mRy between 200 and 300 Ry). Hence:

- r2SCAN, TPSS and other smooth meta-GGAs: `Grid_Spacing` <= 0.16 bohr
  (0.1 mRy, O2 0.35 mRy, levels to 2 meV); 0.13 bohr for energy differences
  to 0.05-0.1 mRy with first-row atoms;
- SCAN: 0.13 bohr or finer, and still up to 0.2 mRy and 25 meV; prefer
  r2SCAN (built to remove this sensitivity) unless SCAN is required;
- forces: the force formulas of the GGAs are the derivatives of the
  meta-GGA's grid energy too (the orbital term does not depend on the atoms;
  a core charge adds its term, see
  [FORCES_AND_RELAXATION.md](FORCES_AND_RELAXATION.md)): finite differences of
  the energy match them as for PBE; against Quantum ESPRESSO (distorted H2O,
  200 Ry) the largest component of r2SCAN differs by 1.5e-3 Ry/bohr at
  h = 0.16 and 5e-4 at 0.13 (PBE: 1.0e-3 and 2.6e-4);
- the LUMOs of small molecules near 0 eV are diffuse and confined by
  `Boundary_Sphere_Radius` (as by a plane-wave box): compare them with a larger
  sphere.

`python -m parsec_python.acceleration.benchmarks.meta_gga_accuracy --input
parsec.in --refine 2` runs an input and estimates the quadrature error of its
meta-GGA energy and levels (the `fft2x` rows minus the `code` rows; the
orbitals interpolated to h/2 on the bounding box, NumPy: about 30 s for H2O at
h = 0.16 bohr).

### Periodic cells

```text
Boundary_Conditions: bulk                          # or slab, wire
Correlation_Type: scan                             # or LibXC_Functional: mgga_x_r2scan mgga_c_r2scan
Kpoint_Method: mp                                  # or a Kpoint_List, or the Gamma point alone
```

Bulk cells with a Monkhorst-Pack mesh, an explicit `Kpoint_List` or the
irreducible wedge of `Symmetrize_Charge_Density`, bulk cells at the Gamma
point, slabs and wires, on both sets of routines (`Use_Fortran_Routines`),
unpolarized and with collinear spin, with forces, relaxation and molecular
dynamics where the semilocal functionals of the cell have them, from both
CLIs ([SCF/meta_gga_periodic.py](SCF/meta_gga_periodic.py),
[Hamiltonian/kpoint_operator.py](Hamiltonian/kpoint_operator.py)). The
k-point workers apply the orbital terms on their GPUs and form `tau` (and `g`)
where the orbitals are; the accelerated Gamma-point path of an orthorhombic
cell holds them in its device Hamiltonian as for a cluster.

- **Bloch factors.** With `psi_nk = e^{ik.r} u_nk` the derivative is
  `(G + ik) u` (`G` wrapped on the periodic axes, zero beyond a slab or wire,
  Cartesian on oblique cells): `tau = sum_k w_k (d/2h^3) sum_n f_nk |(G + ik) u_nk|^2`
  and the Hamiltonian of k-point `k` holds `sum_c (G_c + ik_c)^H diag(vtau) (G_c + ik_c)`,
  Hermitian and with `dE/dn` the exact derivative of the discrete energy with
  respect to `u` (at `k = 0` the term of the Gamma point). In the orbital form
  `g = sum_k w_k (d/h^3) sum_n f_nk 2 Re[conj(u_nk) (G + ik) u_nk]`, in which the
  `ik` part drops, and `2 sum_c [W_c G_c + G_c^T W_c]` is the same at every
  k-point. Time reversal (`u_-k = conj(u_k)`) leaves `tau` and `g` unchanged,
  so a k and -k merged with twice the weight are exact; an irreducible wedge
  is completed by the space-group average of `tau` (a scalar) and of `g` (a
  vector field, `g(r) = C g(r')` with the Cartesian rotation `C` of each
  operation), as of the density.
- **Spin at the Gamma point** runs the k-point spin loop at `k = 0` (the
  Gamma-point driver of periodic spin has no meta-GGA term).
- **Forces.** No term beyond those of a cluster: the periodic and slab forces
  read `dE/dn_core` (`Forces.periodic.meta_gga_core_potential`), and the ionic
  loops carry `vtau` and `W` to the next structure. Against central
  differences of the total energy (Si, one atom displaced, Andrew Grieder's
  routines): the smooth test meta-GGA as PBE (3.4e-4 and 3.5e-4 Ry/bohr at
  h = 0.4); SCAN 1.8e-3 at h = 0.4, 3.8e-4 at 0.3 (PBE 1.7e-4), SCAN's grid
  sensitivity ([FORCES_AND_RELAXATION.md](FORCES_AND_RELAXATION.md)).
- **GPUs.** The device Bloch operator applies the derivatives along the grid
  directions (`G_c = sum_i M_ci L_i`, recovered from `G`; on an oblique cell a
  third of the entries of a Cartesian `G_c`) with the CSR kernel of the
  cluster term on the real view of the complex block, the pointwise part in
  one kernel ([acceleration/Hamiltonian/device_bloch.py](acceleration/Hamiltonian/device_bloch.py),
  [acceleration/Hamiltonian/bloch_meta_gga_device.py](acceleration/Hamiltonian/bloch_meta_gga_device.py)).
  Bulk Si on one A100 (Fortran's grid of 36^3 points at h = 0.20, 16
  columns): one application of the Bloch Hamiltonian takes 1.29 ms, with the
  orbital terms 1.67 ms; the 4x4x4 mesh (8 irreducible k-points) converges in
  11.6 s and 12 iterations with PBE, 17.0 s and 14 with SCAN (`main.py`:
  26.5 and 32.6 s).
- **Filter bounds.** The Bloch operators carry Gershgorin's bound of their
  orbital terms (`spectral_margin`, with `|G_c + ik_c| <= |G_c| + |k_c|`), so
  a `vtau` spike in the vacuum of a slab or wire cannot escape the Chebyshev
  filter: the host eigensolvers add it to the Lanczos bound, the device
  k-point solver takes `max(lanczos, bare + margin)` as the cluster path.
- **Refused** as for clusters, and: the MPI-distributed k-point solver
  (`--mpi`: `tau` needs the orbitals on every grid point), meta-GGA hybrids
  of periodic cells, `Restart_Run` of a periodic cell (`parsec.dat` holds no
  `vtau` or `W`), the atomic `Potential_Extrapolation` of the
  Fortran-derived k-point ionic loop.

Validation ([tests/unit/test_meta_gga_periodic.py](tests/unit/test_meta_gga_periodic.py),
[acceleration/tests/test_meta_gga_periodic_device.py](acceleration/tests/test_meta_gga_periodic_device.py)):
the Bloch terms against their dense matrices and as the derivative of the
energy (3e-13, orthorhombic and Fortran's general fcc grid); Gamma-point loop
= 1x1x1 Bloch loop (bulk 2e-15 Ry; slabs of both routine families and a wire
1e-8); two equal spin channels = the unpolarized loop (1e-10 Ry); device
loops (CuPy and the NumPy double of the device code) = host loop (1e-9 Ry);
accelerated Gamma-point path = reference path (Si8, 7e-15 Ry density form,
9e-13 orbital form); the energy of a converged k-point loop is the functional
of its orbitals. A 16-atom Si supercell at Gamma against the 8-atom cell with
the 2x1x1 mesh differs by -2.4e-4 Ry with SCAN, as with PBE (-3.2e-4): the
finite-difference Bloch kinetic term of the k-point Hamiltonian. The orbital
form also converges what the density form does not (SCAN, an H2 slab with a
2x2 mesh: SRE 3e-4 after 80 iterations against 21 iterations; an H2 wire 80
and 45 iterations against 16 and 20); in bulk Si the two forms agree to
1e-5 Ry.

Against Quantum ESPRESSO 7.5 + libxc 7.0 with the same FHI98PP potential
(bulk Si, a = 10.26 bohr, Gamma-centred meshes, QE at 200 Ry and dual 8,
the references of `examples/3d_meta_gga`; `Expansion_Order` 12),
`E_mGGA - E_PBE` (mRy) and the gap on the mesh (meV), parsec_python minus QE,
at h = 0.20 / 0.16 bohr:

| functional | 1x1x1 | 2x2x2 | 4x4x4 |
|---|---|---|---|
| r2SCAN (libxc) | -0.14 / +0.5, -0.15 / +0.4 | -0.11 / +0.3, -0.10 / +0.3 | -0.09 / +0.3, -0.09 / +0.2 |
| SCAN (native) | -0.51 / -0.3, -0.59 / -0.9 | -0.09 / +2.2, -0.16 / +2.5 | -0.14 / +4.8, -0.11 / +1.3 |
| TPSS (libxc) | see below | -0.10 / +0.4 (h 0.20) | -0.09 / +0.3 (h 0.20) |

QE's own meta-GGA totals move by up to 1.5e-4 Ry and its SCAN gap by a few
meV between 120 and 250 Ry.

**Limitation: TPSS-type functionals at the Gamma point.** In a periodic cell
sampled at the Gamma point alone, every occupied orbital can have a zero
gradient at the same grid points (in Si the bond centres, at the density
maximum), where `tau` and `sigma` vanish together. The TPSS-type
functionals (`z = tau_W/tau`; the density form of the gradient under `auto`)
may then stall or diverge: Si at Gamma stalls at an SRE of 7e-3 at
h = 0.25 bohr and diverges at h = 0.4, whatever `LibXC_Tau_Threshold` and
`LibXC_Enforce_FHC` say. Remedies: a k-mesh (TPSS converges with 2x2x2 and
4x4x4 and matches QE, table above), or `PARSEC_META_GGA_GRADIENT=orbital`
(converges in 125 and 173 iterations at h = 0.25 and 0.16, `E - E_PBE`
within 0.05 and 0.25 mRy of QE, the gap within 26 and 7 meV). SCAN and
r2SCAN (written with `alpha`) are not affected.

### What is refused

When the input is read: meta-GGA hybrids of periodic cells, spin-orbit
coupling, non-collinear magnetism,
FDET, LDA+U, `Energy_Refinement`, `Polarizability`, non-self-consistent runs
(`nscf`), `Output_GW`, band structures and `Restart_Run` (the restart file
holds the potential, not `vtau` and `W`). In GPU runs: `--mpi` (the
MPI SCF, and with it a sector basis shared by several devices), the FP32
filter (`PARSEC_CUPY_MIXED_FILTER=on`) and
`PARSEC_CUPY_RITZ_HANDOVER=triangular`. One process puts each sector on a
device of its node, several sectors on one device where there are fewer
devices than sectors.

### Switches

| variable | default | meaning |
|---|---|---|
| `PARSEC_META_GGA_GRADIENT` | `auto` | `auto`, `orbital` or `density`: the density gradient (see Method) |
| `PARSEC_META_GGA_ANDERSON` | `weighted` | `weighted` or `plain`: the inner products of Anderson's least squares |
| `PARSEC_META_GGA_MIXING` | `joint` | `joint` mixes `(V, vtau[, W])`; `output` takes `vtau_out` unmixed and `linear` mixes `vtau` linearly (density form only) |
| `PARSEC_META_GGA_FILTER_BOUND` | `gershgorin` | the filter's upper bound with the orbital term: `gershgorin` adds the term's Gershgorin bound to the last bound without it (safe; H2O at h = 0.16-0.20 bohr takes 36-42 % more SCF time than without it, k04 none); `targeted` runs a second short Lanczos recurrence from the term's largest rows; `lanczos` leaves the term out (comparisons only; the filter can fail) |
| `PARSEC_META_GGA_DENSITY_CUT` | `off` | `low,high` (bohr^-3): below `low` the energy is the functional's own `alpha = 1` GGA limit `e(n, sigma, tau_W + tau_unif)`, between `low` and `high` a quintic switch in `log n`, with exact derivatives |
| `PARSEC_SCAN_TAU_FLOOR` | - | `weizsaecker` floors `tau` at `tau_W` for the native SCAN |

## Spin polarization

`Spin_Polarization: true` solves an isolated system with two collinear spin
channels, as PARSEC does: one potential and one eigenproblem per channel,
one Fermi level, and the exchange-correlation functional of the two spin
densities. Without a `Correlation_Type` line the functional is the
Perdew-Wang LSDA (`pl`), as in PARSEC; `ca`, `pbe`, `pw`, `lb` and `cs` have
spin forms too, `xa`, `wi` and `hl` are refused.

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
SciPy, native and CuPy backends. The multi-node runner refuses a
spin-polarized input unless it runs in one process (`--serial-control`). A
spin-polarized periodic cell runs the periodic spin drivers of the readable
solver (see "Physical scope").

### Spin channels in the symmetry sectors

```text
Spin_Symmetry_Sectors: auto      # or off
```

decides whether `main.py` on the CuPy backend solves the two channels in the
symmetry sectors of the point group of the atoms (`auto`), as it solves an
unpolarized run, or on the full grid (`off`). The default is `auto` for every
`Correlation_Type` (until 7 November 2026 it was `off` for the semilocal
functionals; `off` gives their earlier results bit for bit). With
`auto` each channel has its own subspace in every sector, the channel and
total densities, the Hartree potential (solved on the symmetry wedge), the
spin functional (on the device for PBE, PW91, PW92, LB94, Casida-Salahub and
the semilocal part of a hybrid; expanded to the full grid on the host for the
others), the mixer, the residual norms and the energies keep one value per
symmetry orbit, and a hybrid's exchange operator is that of each channel in
every sector. Energies, levels and forces are those of the full grid
(differences within the SCF tolerance: `1e-13 Ry` in the energy of the test
cases; over the spin-polarized inputs of the repository at most `1e-9 Ry` in
the energy and `1e-6 Ry` in an occupied level of a semilocal run, `3e-9 Ry`
for the PBE0 example of O2 at its Fock-loop tolerance). The sectors pay from
some 2e5 grid points on: O2 with PBE on one A100, 1.41 M points, 169 s on the
full grid and 19 s in the sectors; 46 k points, 5.9 s and 7.2 s (the
eigensolver of the sectors does more, smaller solves).

The sectors cannot represent a density that breaks the point group of the
sectors, and they hold a state that breaks the larger point group of the atoms
only in the directions that group allows. A partly filled degenerate level
breaks it on the full grid: the oxygen atom with one minority p electron in
three p levels (at 80 K the sectors keep the symmetric ensemble, 68 mRy above;
at integer occupations an axis-aligned p orbital, 14 mRy above on a coarse
grid), an open 3d shell (Sc 14.8 mRy, V 6.2 mRy above the full grid with PBE0
at h 0.30). The sectors can only be wrong where degenerate partners lie in
different sectors (partners within one sector mix there as on the full grid).
A multiplet is therefore a run of levels of one channel whose neighbours lie
within 0.01 Ry, whose whole spread is at most 0.03 Ry (a longer run is a band
of levels, such as the valence band of a large cluster or the unconverged top
of a first subspace) and whose levels lie in two or more sectors. A sector run
goes to the full grid when one of three tests fails: (1) the start Hamiltonian
(first SCF iteration, from the atomic densities, which has the full point
group of the atoms) has a partly filled multiplet in a channel, `n` electrons
in `d` levels with `min(n, d - n) > 0.05` (a partly filled shell holds a
third of an electron or more per level, a Fermi-Dirac tail of empty levels a
few 1e-3), under every occupation rule
(Fermi-Dirac, `Fermi_Temp: 0`, `occup.in`; an `occup.in` that fills all levels
of a multiplet equally is kept), once the levels are converged (the start
Hamiltonian is solved again from its subspaces, up to four times, until the
levels up to the Fermi level move by at most 1e-4 Ry; a report on the
unconverged first levels of a large cluster disappears or keeps moving and is
left to the tests at the end); (2) at the end of a loop without an exchange
operator or of the run, a channel density differs by 0.01 electrons or more
from its image under an operation of the point group of the atoms that the
grid keeps (a signed permutation of the axes, also outside the sectors' group;
a domain or grid shift with less symmetry than the atoms keeps a subgroup,
which is what is tested). The sectors enforce only the abelian subgroup, so
the other operations also show the SCF error of a symmetric state (C185H124
and C459H204: 4e-3 and 3e-3 electrons at a criterion of 2e-4 Ry, 2e-2 and 1e-2
at 2e-3 Ry, the first steps of a relaxation with `Relaxation_SCF_Upscale`), while
a broken state differs by about an electron (V 1.25, Co 1.17); a converged loop
whose asymmetry lies between 0.01 and 0.2 electrons therefore goes on to a ten
times tighter criterion (at most three times) and measures again, and only an
asymmetry that does not shrink to half leaves the sectors; (3) with
Fermi-Dirac occupations, a level of a
multiplet (as above, or levels equal to 1e-4 Ry in two or more sectors) holds
a fractional occupation at the end, `min(f, 1-f) >= 0.05`. Then `parsec.out`
says so (a `NOTE` with the levels or the operation) and the calculation starts
again on the full grid: from the atomic densities, as with `off`, bit for bit;
in a later step of a relaxation from the potentials of the previous structure,
as an `off` relaxation, with new subspaces (equal to `off` within the SCF
tolerance). Forces, outputs and the next steps of a relaxation then use the
full grid. `auto` keeps the full grid, and says why in `orbital_symmetry`,
without a point group, with `--symmetry off`, on a CPU backend, with LDA+U,
`Symmetrize_Charge_Density`, `PARSEC_CUPY_RITZ_HANDOVER=triangular` or
`PARSEC_CUPY_DISTRIBUTED_FILTER=1`. A restart (`Restart_Run`) of a
spin-polarized run solves both channels on the full grid (a `NOTE` says so),
and with `--symmetry on` a run that leaves the sectors drops the forced
symmetry for the full grid. What the tests cannot see: a partly filled
multiplet whose partners no grid operation relates (the `e` levels of a
threefold or sixfold axis) and that is not partly filled at the start; one
that the first filter of a large system leaves unresolved and whose sector
state ends invariant and integer; a state that passes through a partly filled
multiplet during the SCF and returns to an invariant integer one.
The reference solver (`reference_main.py`) and the MPI runner keep the
full grid in every case; without `Spin_Polarization` the keyword is ignored
with a WARNING. `Generate_Complete_Parsec_In` writes it for a spin-polarized hybrid
and wherever the input sets it to `off`.

With `ca`, energies, levels and forces agree with Fortran PARSEC to the
digits it prints, and so do `pl`, `lb` and `cs`. With `pbe` and `pw` they
differ as they do without spin, because this package uses the potential of
the discrete energy. A state with an
empty spin channel (a hydrogen atom, any one-electron system) is an ordinary
calculation here; in Fortran PARSEC it does not converge with `ca` and loses
its exchange energy with `pbe`. Since 5 November 2026 this holds in every driver:
the spin forms of periodic cells, spin-orbit and non-collinear runs
(`ca_lda_spin_polarized`, `pbe_spin_polarized`) had kept PARSEC's rule of
evaluating only where both channels are positive (CA: no exchange-correlation
at all when a channel empties) or a 1e-16 floor (PBE: potentials of 1e7 Ry in
the empty channel); they now take the fully polarized limit wherever the
total density is positive, and points with both channels positive keep their
values. Two things are not unique in either program:
the sign of the moment, and the orientation of a partly filled degenerate
shell such as the oxygen atom at the default temperature, for which
occupations fixed in `occup.in` give a reproducible state. Details:
[`PYTHON_IMPLEMENTATION.md`](PYTHON_IMPLEMENTATION.md#spin-polarized-calculations)
and [`PARSEC_ALGORITHM.md`](PARSEC_ALGORITHM.md#spin-polarized-calculations).

## LDA+U and the TS dispersion correction

`ldaplusu_u` and `ldaplusu_j` (per `Atom_Type`, an energy, default unit Ry)
add PARSEC's on-site term of the d channel (`upot.F90`): five projectors per
atom, the radial pseudo-wavefunction of the d channel of the file times the
real d harmonics, cut at the radius that holds 95 % of its norm and not
renormalized. From the second SCF iteration on, the occupation matrices of
the orbitals of the iteration before give the potential matrix
`D = -U N + 2J (P3:N + P4:N') + (U-J)/2` and the energy `LDA+U Energy` that
is added to the electronic energy; `parsec.out` shows the two terms of
`calcden.f90z` every iteration, as PARSEC does. A species whose file has no
d channel keeps no term (a warning says so, PARSEC ignores it silently). The
n-th `ldaplusu_u` line goes to the n-th `Atom_Type`, as ESDF hands it out;
a line that would land on another species than the one it stands under is
refused.

```text
Atom_Type: Zn
Local_Component: s
ldaplusu_u 5.0 eV
```

The term runs in the scalar loops (isolated, isolated spin-polarized,
periodic Gamma-point and periodic spin-polarized cells), in the scalar and
collinear-spin k-point loops with Bloch projectors (`SCF/kpoint_hubbard.py`;
PARSEC's own k-point form is only right at k = 0), in the band structures of
their converged Hamiltonians and with `Energy_Refinement`. `main.py` runs a
cluster with `ldaplusu_j = 0`, unpolarized or spin-polarized, on the
accelerated driver, with the term in its device Hamiltonians (also in the
symmetry sectors and on several GPUs) and its occupation matrices and forces
formed where the orbitals are (`acceleration/README.md`, "LDA+U and the TS
correction"), and so it runs a bulk cell at the Gamma point, with its forces,
relaxations and dynamics. Its k-point and periodic spin runs go to the
reference solver, whose k-point workers apply the term in their device Bloch
operators and form the occupation matrices from their device orbitals when a
GPU is visible; so do its band structures and the Gamma-point spin runs of a
wire or slab (the 1x1x1 k-point driver). A `J` term stays on the reference
solver's CPU path, and the other wire and slab runs on the reference solver.
A wire or slab has it in the
same drivers, its Bloch projectors replicated along the periodic directions
alone (`nreplica(1:per)` of `upot.F90`) and the occupations of an irreducible
wedge averaged over its group of `alatt` (PARSEC's k-point LDA+U of a wire
stops in `upot`). Spin-orbit coupling and non-collinear magnetism are
refused with it (PARSEC's spinor path is not a spinor LDA+U). Forces,
relaxation and molecular dynamics work with `ldaplusu_j = 0` and without
k-points, with the derivative of the projectors (PARSEC's own LDA+U forces
are zero, see `forcnloc_u`); with `ldaplusu_j` or k-points they are refused,
because PARSEC's J term (its P3/P4 tensors applied to the real d harmonics)
is not the derivative of its energy and the Bloch projector gradients are
not written.

`Van_Der_Waals: true` adds the Tkatchenko--Scheffler energy of the converged
density (`efvdw.f90`) to the total energy of the report after convergence
(`TS-vDW correction (included)`) and its forces to the forces, after every
SCF driver (k-points, periodic spin, spin-orbit and non-collinear loops too;
not for wires and slabs, where PARSEC needs a bulk cell). It needs
`Correlation_Type: pbe` and `Ignore_Symmetry: true`, as in PARSEC; a periodic
cell reads `Hf_One`, `Hf_Two` and `Hf_Three`, the Hirshfeld cells. The
Hirshfeld volumes are those of the structure of each SCF loop (PARSEC keeps
those of the input structure). On a GPU `main.py` forms the Hirshfeld sums
and the pair sum on the device: of a cluster and of a bulk cell at the Gamma
point on the accelerated driver (single points, forces, relaxations and
dynamics), and of a bulk cell in every periodic driver
of the reference solver whose eigenproblems run on the GPU (with its
Hirshfeld cells and lattice shells); a cluster on the reference solver keeps
the host sums.

## Frozen-density embedding (FDET) and AFM scans

PARSEC 1.5's FDET embeds an AFM tip (solved by Kohn-Sham) in a frozen sample
that an earlier run wrote to `pot.dat` (`Embedding/`, `Output/pot_dat.py`).
The sample run gives `Write_Potential_Field: true` (a PARSEC.py switch; PARSEC
1.5 writes `pot.dat` after every run, this port only on request), the tip run
`Potential_Field: true` with `Potential_Field_Name` (default `pot.dat`), the
sample ions as point charges (`Add_Point_Charges`, `Pt_Chg` = valence charge,
sample-frame coordinates), the `Boundary_Sphere_Radius` and `Grid_Spacing` of
the sample run (classical mode: the tip on the sample grid) or
`Boundary_Sphere_Around_Tip: true` with a small sphere of its own (needs
`Ignore_Symmetry: true`). `Kinetic_Energy_Functional` is `tf` (default), `tw`,
`pw` (PW91k), `pb` (PBE-like, `pbe_m`/`pbe_k`) or `no`. Two defects of the 1.5
potentials are corrected by default and kept for comparisons with
`FDET_Kinetic_Form: parsec1.5`: the `pw`/`pb` potentials of 1.5 use twice the
`u` of their energy derivative, and 1.5's `tw` is TF + vW/18 instead of the
gradient expansion TF + vW/9. The total energy is 1.5's FDET energy (with the
tip sphere shifted by a constant of the sample: only differences count);
forces keep the net force, as in 1.5. A tip run is never recentred.
`Molecular_Dynamics` (classical mode) integrates the motion of the centre of
mass that this net force drives; PARSEC 1.5 removes the centre-of-mass
velocity in every step also here, which throws away the work of the net force
(the energy is then not conserved).

```text
Potential_Field            true
Potential_Field_Name       sample_pot.dat
Kinetic_Energy_Functional  pw
Boundary_Sphere_Around_Tip true
AFM_Scan                   true
AFM_Scan_Heights           3
AFM_Scan_Height_Step       0.4 bohr
begin AFM_Scan_Lateral_Points
 0.0 0.0
 0.8 0.0
end AFM_Scan_Lateral_Points
```

`AFM_Scan` (PARSEC.py; 1.5 leaves scans to the closed PARSEC-Edge app) moves
the tip rigidly over the lateral points and heights (the input height in the
middle) as one manual sequence with warm starts and writes `afm_scan.dat`
with `E(x, y, z)` and `k_ts = d2E/dz2` by three-point differences (Ry/bohr^2
and N/m; no tilt correction). The table is rewritten after every tip position,
so a run that stops keeps the points before; a `k_ts` whose three SCF loops
did not all converge is left out (`unconv`). Every tip position is checked
against the domain when the input is read. `AFM_Scan_Line_Range: first last`
runs a part of the lines, so that several processes or GPUs share a scan
(table `afm_scan_lines_<first>-<last>.dat`). Each part needs a folder of its
own: the parts write the same `parsec.out`, archive, `atom.cor` and restart
files. The first scan in a folder writes `afm_scan_part.lock` with its line
range, and a part with another range is refused there. `Minimization:
manual` scans of `manual.dat` work as in 1.5, with 1.5's warm starts (the
tip-sphere field moved under the kept orbitals; in the classical mode density
and orbitals translated when the tip moves by whole grid steps). Spin,
periodic tip runs (1.5's slab samples), symmetry and MPI are refused with
FDET. Both `main.py` (accelerated driver) and `reference_main.py` run it.

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

`Pseudopotential_Format: martins_wang` reads `<Atom_Type>_POTRW.DAT`, which
names its channels, occupations, ionic charge and local channel itself: the
input gives only `Core_Cutoff_Radius` and no `Local_Component`.
`Pseudopotential_Format: fhipp` reads an FHI98PP `<Atom_Type>_FHIPP.DAT`
with `Potential_Num`, `Local_Component`, `Core_Cutoff_Radius`,
`Electron_Per_Orbital` and, for its core density,
`Nonlinear_Core_Correction: .true.`. As in Fortran PARSEC, the n-th
occurrence of such a label goes to the n-th Atom_Type that reads it;
`Potential_Num` is read for `martins_old` and `fhipp` species only, as PARSEC
1.5 reads it (the PARSEC before 1.5 read one for every species: such an input
is refused with a message).

The local potential of an FHIPP, MARTINS_OLD or MARTINS_WANG table reaches the
Coulomb tail `-2Z/r` only to the digits of its file (`r V + 2Z` of 1e-13 to
1e-11 relative for FHIPP, out to the end of the table). In a periodic cell its
images are summed out to the radius where the table is `-2Z/r` to that
precision (6 bohr for silicon), as for a POTRE table. Before 2026-11-07 they
were summed to the end of the table (75 bohr). With `Cubic_Spline` the error of
the spline there then lowered the local potential of orthorhombic cells by
0.05 Ry (an 8-atom silicon cube 1.62 Ry low in total energy), and preparing
a 64-atom cell took 11 minutes (now under 3 s). Oblique cells (Fortran's
general grid) with `Cubic_Spline` species take the same sums since then; their
earlier direct sum met the same spline error as the cell grew (a 54-atom fcc
silicon cell 0.049 Ry low in the potential), and primitive fcc cells move by
less than 5e-5 Ry (`PARSEC_PERIODIC_OBLIQUE_SPLIT=readable` restores the
earlier sum; acceleration/README.md has the numbers).

`Pseudopot_File` (PARSEC 1.5, one per Atom_Type, the n-th to the n-th) names
the file of a species instead of `<Atom_Type>_<suffix>.DAT`: a relative path
is looked for beside the input, then in `--pp-dir` and the current directory.

Pseudopotentials and the selected XC functional must be consistent: a file
generated for another functional is refused with `ca` and `pbe`, and used
with a warning with the other functionals, as Fortran PARSEC warns; a CA
file in a PBE run is used with that warning too (PARSEC 1.5's `Tier_xc`
GGA tests are such runs). The
solver can read converted norm-conserving UPF data, but conversion does not by
itself validate transferability, ghost states, relativistic content, or
core-hole suitability.

## PARSEC 1.5 input keywords

The reader takes the keywords that PARSEC 1.5 (`usrinputfile.F90` of the 1.5 package) added to the input and I/O
layer, with 1.5's validation (`Input/parsec_input.py`, block `_INPUT15_*`):

- `Mol_RNG_Seed` is 1.5's name of `RNG_Seed` and `Npcol` that of `MPI_Groups`; both names are read, an input with
  both is refused.
- Keywords that only tune 1.5's MPI layout and libraries are read with their type and change nothing (one warning
  names them): `Use_ScaLAPACK`, `ScaLAPACK_Block_Size`, `Use_ELPA`, `ELPA_Block_Size`, `Npcol` (1.5 also rounds
  `States_Num` up to a multiple of `Npcol*ScaLAPACK_Block_Size`; this run keeps the input's, the warning gives 1.5's
  number), `Nslices` and the spectrum-slicing keywords (`Spectrum_Slicing: .true.` runs the input's eigensolver, which
  converges to the same states), `Use_Sfc`, `Thread_Random_Init`, `Rritz_Logic`, `Test_Logic` (`none` only),
  `Max_Taskgroups`, `Vxc_Matrix_Elements`.  `Use_HDF5: .true.` writes no HDF5 file: `parsec.dat` is written as with
  `.false.` (a 1.5 binary without HDF5 writes neither).
- `Use_LibXC: .true.` with `Func_Id_X`/`Func_Id_C` runs libxc's functionals (the optional library; see
  [Functionals of libxc](#functionals-of-libxc)): 1.5's defaults and its 0 for an empty slot, from the first iteration
  (1.5 starts every libxc run with PBE) and with the exact exchange of a libxc hybrid (which 1.5 leaves out); with
  `.false.` (1.5's default) `Func_Id_X`/`Func_Id_C` are ignored, as 1.5 ignores them there.
- `Correlation_Type: wb` (1.5's PBE with the White-Bird variational potential) runs this port's PBE, whose potential
  is the exact derivative of the discrete energy, as 1.5's `wb` of an unpolarized cluster without symmetry (H2O:
  -34.56585964 Ry here, -34.56585973 Ry with 1.5 `wb`); 1.5's spin `wb` is not the derivative of its energy.
- `Hartree_CG_Tolerance` / `Hartree_CG_Criterion`: without them the Poisson CG stops at 1e-7 of the initial residual
  within 1600 products (this port's default, kept on 2026-11-04); with either of them it is 1.5's CG: the
  given relative tolerance (1.5's default 1e-13), the test against the initial residual (1) or the right-hand side
  (2), 2400 products.  Criterion 2 of a cluster runs on the reference solver.
- `Generate_Complete_Parsec_In: .true.` writes `bak_complete_parsec.in` beside the input: the effective input of the
  run in atomic units, the cell and atoms after `Super_Cell`, which read again gives the same problem
  (`Input/complete_input.py` makes the text, `Output/complete_input.py` writes the file).
- `Output_Ascii_Data` and `Print_Unconverged_Density` (`.true.`) are refused with the reason (debug dumps of 1.5's
  per-rank layout; cube files and `Save_Intermediate_Charge_Density` hold the same fields).
- `Periodic_System` is never read by PARSEC: without the line it follows `Boundary_Conditions`.
- Warnings where 1.5 would stop or warn: `States_Num` below `N_val/2 + 2` (1.5's ierr 213), a CA pseudopotential in a
  PBE run.  `Fermi_Temp` takes ESDF's energy units (`0.05 eV` is 580.226 K, as in 1.5).

Eigensolver and SCF controls (`Input/solver_options.py`, measured in the 1.5 round):

- `FF_MaxIter` (CHEBFF first-filter cycles) defaults to 3, PARSEC 1.5's default, since 4 November 2026 (2 before;
  spinor runs, non-collinear or with spin-orbit coupling, take 3 too but keep the degree decrease after the fifth
  iteration: with the decrease after the fourth the warm-started field loops of a spin-orbit polarizability run need
  as many steps as the first;
  the default chosen after an A100 batch on the nanodiamonds k06-k14: SCF 7-13 % shorter than with 4, equal
  iterations, energies within 1.1e-4 Ry). 1.5's range 0..50 is accepted and a value outside resets to 3. With 0,
  1.5 runs no first-filter cycle: it keeps the wavefunctions of an HDF5 restart (`restart.h5`, its
  `restart_run_just_check_force` tests), random vectors otherwise; this port reads no `restart.h5`, so 0 runs the
  default 3 cycles, with a warning. `Bands_FF_MaxIter` defaults to `FF_MaxIter`, as in 1.5.
- The later-SCF Chebyshev degree may be lowered by one at the end of every iteration after the fourth (PARSEC 1.5's
  `iter > 4`; before 4 November 2026 after the fifth). `Chebyshev_Degree_Decrease_After: 5` (an extension of this port) gives the earlier rule bit
  for bit; `Convergence_Criterion_Approach` (default 100 x `Convergence_Criterion`; 1.5's is 2 x) sets the SRE below
  which it happens.
- The default eigensolver stays CHEBDAV (1.5's is CHEBFF): CHEBFF is faster on the large clusters, whose inputs
  write it, and slower or less accurate at the same criterion on Si28H36 and on periodic cells.
- Opt-in labels of 1.5: `Vector_Rng_Seed` (start vectors of the first solve; -1 a seed from the clock, named in a
  warning), `Smart_Subspace` (1.5's SUBSPACE lower bound, on the reference and the device filters since 8 November
  2026, see [Bulk cells: the SUBSPACE lower bound](#bulk-cells-the-subspace-lower-bound)), `Chebyshev_Scaled`,
  `Bounds_Method`, `Bounds_Iter` (1.5's scaled filter and Lanczos bounds, on the reference eigensolvers),
  `Filter_Restart_After` with its two companions
  (relaxation and molecular dynamics). Under MPI every rank takes rank 0's clock seed, so that the seed printed
  repeats the run.
  Extensions: `FF_Residual_Test: stop` (1.5's residual-driven CHEBFF at `Diag_Tolerance`, which stops the run as 1.5
  does when `FF_MaxIter` cycles do not reach it; `continue` keeps only the early exit; off by default, so an explicit
  `Diag_Tolerance` keeps its earlier meaning), `Exact_State_Count: .true.` (1.5's working space of exactly
  `States_Num` vectors, which has no buffer) and `Reorthogonalization_Threshold` (0.1; 1.5 uses 0.68).
  `Subspace_Buffer_Size`, which 1.5 no longer reads, keeps its earlier meaning: 0 to 5 reset to 6 with a warning.

`--esdf-lenient` (both command lines) reads an input as ESDF does: labels this reader does not know are ignored and a
label given twice takes its first value, each with a warning.  The inputs of 1.5's test suite need it for lines such as
`boundary_mask false`, `scfso true` or `nscf_kpoints (block)`, which ESDF ignores without a word; by default such
lines are refused.

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
