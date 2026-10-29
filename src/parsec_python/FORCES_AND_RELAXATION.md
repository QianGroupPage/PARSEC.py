# Forces, structural relaxation and molecular dynamics

This document describes what moves the atoms of an isolated system or a periodic cell in
`parsec_python`: the atomic forces, the three minimization schemes of PARSEC
(`Minimization: bfgs | simple | manual`) and its molecular dynamics. It says
how to run them, which Fortran routine each part follows, where the package
deliberately differs, and what was compared with Fortran PARSEC.

Units are PARSEC's throughout: bohr, Rydberg, Ry/bohr.

## Scope

- Isolated systems (`Boundary_Conditions: cluster`), sphere or box domain,
  CA/PZ LDA or PBE, without or with spin polarization, with or without
  nonlinear core correction, projectors through `l = 3`.
- Periodic cells (`Boundary_Conditions: bulk`), without spin polarization and without spin-orbit coupling, at the Gamma
  point or with a Monkhorst-Pack k-point sampling: forces, `Minimization: bfgs | simple | manual` and molecular dynamics.
  See "Periodic cells" below. A spin-polarized or spin-orbit periodic calculation, and the accelerated (GPU) path, do not
  have them yet.
- A spin-polarized calculation uses the same routines: `forcloc.f90` reads
  the total density and the averages over the two channels of the potential
  difference and of the exchange-correlation potential, and `forcnloc.f90`
  sums over the states of both channels with the weight `4/nspin`.
- Not translated from `forcloc.f90` and `forcnloc.f90`: point charges,
  LDA+U, van der Waals corrections, spin-orbit and non-collinear terms.
- Every execution path computes the same numbers: the readable NumPy path
  (`reference_main.py`), the accelerated single-node path (`main.py`, CPU or
  GPUs) and the multi-node GPU runner
  (`acceleration/benchmarks/mpi_full_scf.py`).

## Running a calculation

### Input keywords

| Keyword | Default | Meaning |
|---|---|---|
| `Skip_Force` | absent | `false` adds the forces to a single point. A relaxation or dynamics run computes them regardless. A single point without the line computes none, so existing inputs keep their cost. |
| `Minimization` | `none` | `bfgs` (limited-memory BFGS), `simple` (steepest descent), `manual` (structures from `manual.dat`). |
| `Movement_Num` | 500 | Largest number of movements `M`: at most `M + 1` structures are computed. |
| `Force_Min` | 0.01 Ry/bohr | The run ends when the largest force on a movable atom is below it. |
| `Max_Step` | 0.05 bohr | `simple`: largest displacement of a coordinate per movement. `bfgs`: when given, every coordinate stays within it of its initial value for the whole run; absent, BFGS is unbounded. |
| `Min_Step` | 1e-4 bohr | `simple`: displacement of a coordinate whose force equals `Force_Min`. |
| `BFGS_number_corr` | 7 | Corrections the limited-memory matrix keeps. |
| `Relax_Restart` | `false` | Read `relax_restart.dat` beside the input: its coordinates replace those of the input and its BFGS block continues the optimizer. |
| `Move_Flag` (per `Atom_Type`) | 0 | 0 moves every atom of the type, `n > 0` its first `n` atoms, a negative value reads a fourth column of `Atom_Coord` (1 moves, 0 fixes). |
| `Molecular_Dynamics` | `false` | Switches molecular dynamics on; it excludes `Minimization`. |
| `Cooling_Method`, `Tinit`, `T_Final`, `T_Step`, `Time_Step`, `Friction_Coefficient`, `Step_num`, `Restart_mode`, `rng_seed`, `test_rng` | PARSEC's | Controls of `moldyn.f90`; see "Molecular dynamics" below. |
| `Initial_Structure` | `input` | Not a PARSEC keyword. `mlff` relaxes the input structure with a machine-learned force field (UMA, MACE, `MLFF_Model`) before the first SCF loop; see [MLFF/README.md](MLFF/README.md). |
| `Potential_Extrapolation` | `none` | Not a PARSEC keyword. How the SCF loop of a moved structure starts: `none` as PARSEC, `atomic` with the superposed atomic densities moved along with the atoms, `second_order` with the density change also extrapolated along the trajectory; see "Start of the next SCF loop" below. |
| `Relaxation_SCF_Upscale` | 1 | Not a PARSEC keyword. `bfgs` and `simple` only: the SCF loop of a structure may stop at up to this many times `Convergence_Criterion` while the forces are large; 1 keeps the criterion fixed as PARSEC does. See "SCF criterion of a relaxation". |

### Command lines

```powershell
python src\parsec_python\main.py calculation\parsec.in
python src\parsec_python\main.py calculation\parsec.in --restart-from previous\relax_restart.dat
python src\parsec_python\reference_main.py calculation\parsec.in
```

`--restart-from FILE` starts from the structure of another run without
editing the input: a `relax_restart.dat` written by this package or by
Fortran PARSEC (its BFGS block continues the optimizer at the coordinates of
the file), or an XYZ file of any program, whose last frame gives the
coordinates in Angstrom; they are recentred like those of an input and the
optimizer starts afresh. The file must list the atoms in the order of the
input, type by type; another order is refused. On several GPU nodes the launch script of the
repository takes the same input:

```bash
scripts/run_multi_gpu_scf.sh calculation/parsec.in output_directory
```

The exit status is 0 for a finished run, 3 when an SCF loop did not converge
and 4 when the structure is not relaxed (movements used up, or the optimizer
stopped above `Force_Min`).

Two inputs of Fortran PARSEC are kept as runnable examples with their Fortran
trajectories: `examples/0d_water_geoopt` (BFGS) and
`examples/0d_SiNa4_relaxation` (BFGS with an old-format pseudopotential and
occupations from `occup.in`).

### Files

A relaxation writes beside the input (the multi-node runner: into its output
directory):

| File | Content |
|---|---|
| `parsec.out` | Every SCF loop, the force tables of every structure, the lines of the mover, and at the end a summary with the wall time of the parts of the ionic steps. |
| `atom.cor` | PARSEC's file: every structure with its energy per atom and largest force. |
| `relax_restart.dat` | PARSEC's file, rewritten after every movement: the next structure and, for BFGS, the complete optimizer state. It is exchanged with Fortran PARSEC in both directions. |
| `iterate.dat` | The iteration table of L-BFGS-B, as PARSEC writes it. |
| `relaxation.xyz` | Extended XYZ: every structure in Angstrom with its energy (eV and Ry) and forces (eV/Angstrom). |
| `parsec_python_results.npz` | Unless `--no-archive`: the result of the last structure and, under `relaxation_*`, positions, energies and forces of all of them. |

Molecular dynamics writes PARSEC's `md_nrg.dat`, `md_mech.dat`, `mdinit.dat`,
`atom.cor` and `movie.xyz`.

### From Python

```python
from parsec_python import ionic_backend, prepare_single_point, run_relaxation
from parsec_python.Input import parse_parsec_input

translation = parse_parsec_input("calculation/parsec.in")
system = prepare_single_point(translation.problem)
outcome = run_relaxation(
    system,
    translation.relaxation,
    ionic_backend(),
    movable=translation.movable,
)
print(outcome.records[-1].energy, outcome.records[-1].maximum_force)
```

These are the accelerated workflow functions, as elsewhere at the top level
of the package; `prepare_reference_single_point` with
`reference_ionic_backend()` run the readable path. `compute_forces(system,
result)` and `compute_reference_forces(system, result)` give the forces of a
converged single point. A system prepared with symmetry holds for structures
of its point group: moving the atoms off it stops with an error, so a
relaxation whose movable flags break the symmetry is prepared with
`symmetry="off"`, as the command line does.

## Forces

The order is that of `parsec.F90`.

1. **Ion-ion** (`forceion.f90`, `Forces/ion_ion.py`):
   `F_a = sum_b 2 Z_a Z_b (R_a - R_b) / |R_a - R_b|^3`.
2. **Local pseudopotential** (`forcloc.f90`, `Forces/local.py`): three grid
   sums with `d = r - R_a`,
   - ionic: `integral rho(r) V_local'(|d|) d/|d|`;
   - SCF correction: `integral (V_out - V_in)(r) rho_atom'(|d|) d/|d|`, which
     removes the first-order error of an SCF loop that stopped at its
     criterion;
   - core correction: `integral V_xc(r) rho_core'(|d|) d/|d|` for species with
     a nonlinear core correction.
3. **Nonlocal projectors** (`forcnloc.f90`, `Forces/nonlocal_force.py`):
   `F_a = 4 sum_n f_n sum_(q on a) s_q <beta_q|psi_n> <grad beta_q|psi_n>`.
4. **Symmetrization** (`symm_forces.f90`, `Forces/symmetry.py`) over the
   cubic point-group operations (signed permutations of the axes) that map
   the atoms onto themselves, then removal of the net force.

The radial derivatives are the tables of `pseudo.f90`
(`Pseudopotential/radial_derivatives.py`): an 18th-order central difference
on the logarithmic grid, the local potential tabulated as `r^2 dV/dr` with
the tail `2Z/r^2`, and the projector derivative held constant near the
origin. Grid points closer to an atom than its first radial sample are left
out, as in PARSEC.

`parsec.out` prints the tables PARSEC prints: the ion-ion, local, SCF
correction, core correction and nonlocal forces, the total before and after
symmetrization, the net force that was removed and the torque.

## Periodic cells

A periodic cell has forces, relaxation and molecular dynamics through the periodic backend
(`Relaxation.periodic_backend`) of the readable solver. `main.py` hands such an input to `parsec_python.cli`
(`NOTE: ... running the reference parsec_python.cli solver instead`): the accelerated path moves isolated structures only.

Forces (`Forces.compute_periodic_forces`; the Fortran routines are `forpbc.f90`, `force_loc_pb.f90`, `ewald_sum.f90` and
`forcnloc.f90`):

1. **Ion-ion**: the exact derivative of the Ewald energy (`Forces/ion_ion.py: ewald_ion_ion_forces`): the real-space sum
   over images and the reciprocal sum.
2. **Local pseudopotential** (`Forces/periodic_local.py`): `integral rho(r) grad V_a(r - R_a)` over the images of the local
   potential as it is built (the short-ranged `V_a(d) + 2 Z erf(eta d)/d` and the reciprocal series), and
   `integral V_xc(r) rho_core'(d) d/|d|` over the images of the core density. There is no SCF-residual term, as in
   Fortran's periodic path.
3. **Nonlocal projectors** (`Forces/nonlocal_force.py`): the same formula with the Bloch projectors of the k-point,
   `F_a = 4 sum_k w_k sum_n f_nk sum_q s_q Re[conj(<beta_q|u_nk>) <grad beta_q|u_nk>]`. A k-point run keeps no eigenvectors,
   so `run_scf_kpoints(..., compute_forces=True)` accumulates the term from them at the end of the SCF loop
   (`result.nonlocal_forces`). The row-distributed MPI k-point solver does not support it.
4. The **net force is removed**, as Fortran does for a crystal. The forces are **not symmetrized** (no space-group
   operations are found) and no torque is reported; the force report prints `(Ewald forces included)` as Fortran's.

The terms are exact derivatives of the energies the SCF evaluates (`Forces` tests: finite differences of each energy and of the
SCF total energy, 1e-5 Ry/bohr or better), with one exception: the local potential and the nonlocal projectors use PARSEC's
derivative tables, which differ from the exact derivative of their linear interpolation by about 1e-4 Ry/bohr in the
test systems.

Atoms leave the cell as they move. The coordinates of the optimizer and the integrator are kept continuous (they are never
wrapped), and every image sum is built for the atoms folded into the cell (`PeriodicPreparedSinglePointSystem.cell_atoms`).
Fortran's `back_to_cell` wraps the positions in the state; the physics is the same, the positions in `atom.cor` and
`movie.xyz` are the unwrapped ones.

A move rebuilds the local potential, the projectors, the densities and the Ewald energy for the new positions
(`SCF.update_periodic_geometry`) and the SCF loop continues from the last mixed potential. The Gamma-point loop also carries
its eigensubspace; the k-point loops start their eigenproblems cold (`Potential_Extrapolation` applies to both). The image
sums stop where the tables end: the local potential tables are `-2Z/r` to 1e-14 Ry beyond 2 to 7 bohr and the atomic
densities are below 1e-16 beyond 12 (core) to 35 (valence) bohr, so a move of a small cell takes a fraction of a second
instead of the minute the 118-bohr tables needed.

`Cubic_Spline` does not change the periodic local potential, as Fortran's: there the potential is a sum of form factors
that integrate the table to high order, so it is the accurate potential whatever the flag says. The periodic local
potential and its force therefore always use the cubic spline of the table (`V_ion.ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE`
set to `False` gives linear interpolation). Linear interpolation of the logarithmic table (1 % steps) shifted the Ge3
total energy by +0.6 to +0.9 mRy against Fortran at every grid from 24 points per side up; with the spline the difference
is 0.003 to 0.09 mRy, the remainder being Fortran's own finite-difference estimate of the `alpha` energy (+0.064 mRy) and,
on a coarse grid, aliasing. The projectors and the core density follow `Cubic_Spline`.

**Grid of an input file.** An input file gets the Fortran grid: half a step off the origin along vectors orthogonal to the
others (zero with `Ignore_Symmetry` and for non-orthogonal vectors) and the largest even number of points per vector whose
step is not below `Grid_Spacing` (`adjust_step`; steps may differ between the axes of a rectangular cell). The
`PeriodicGridSettings` of the API keep `shift=0` and `points_rule="round"`.

**A caveat for hard pseudopotentials.** On a coarse grid the energy of a hard pseudopotential (the hydrogen of
`examples/0d_AuH` at 0.6 bohr) changes by Ry with the offset of the grid, and the energy of a rigid translation by tens of
mRy at 0.3 bohr. The forces are consistent with that energy (net force removed, as in Fortran, but then a relaxation can
slide a molecule over the grid) and Fortran's reciprocal-space potential has much less of it.

Validated against Fortran PARSEC (`parsec_tests/pbc_forces`; three Ge atoms with a nonlinear core correction, forces up to
2.5 Ry/bohr, the same grid in both codes). Largest difference of the total force after the net force is removed, and of the
nonlocal term, in Ry/bohr, with the total-energy difference (Python minus Fortran) in Ry:

| cell | k-points | points per side | total force | nonlocal | energy |
|---|---|---|---:|---:|---:|
| cubic, 9.6 bohr | Gamma | 24 / 32 / 40 | 2.0e-3 / 6.1e-4 / 4.1e-4 | 3e-5 / 6e-6 / 3e-6 | -4.5e-5 / +8.5e-5 / +7.2e-5 |
| cubic, 9.6 bohr | 2, generic | 24 | 2.0e-3 | 3.4e-5 | -9.3e-6 |
| fcc primitive, 10.2 bohr | Gamma | 20 / 26 / 32 / 40 | 3.3e-4 / 2.7e-4 / 2.3e-4 / 1.5e-4 | 1.5e-5 / 4e-6 / 9e-7 / 9e-7 | +2.2e-4 / +1.9e-4 / +2.0e-4 / +2.0e-4 |
| fcc primitive, 10.2 bohr | 2, generic | 20 | 2.4e-4 | 8.7e-6 | +2.1e-4 |

The k-point forces are as close to Fortran as the Gamma ones at the same grid, and the oblique cell is as close as the cubic
one. The energy offset is Fortran's: its `alpha` energy is a finite difference in q (spacing 0.015) and is low by 6.4e-5 Ry
for the 885 bohr^3 cubic cell, by 2.1e-4 Ry for the 265 bohr^3 fcc cell (it scales as 1/volume), which is the whole offset
at 32 points and above. The residual force difference falls with the grid: Fortran's local and core forces are reciprocal
sums over form factors (they equal ion-ion plus the numerical derivative of that functional to 2e-5 on PARSEC.py's density),
PARSEC.py's are the exact derivatives of the grid-sampled energy. The relaxation and dynamics loops are tested on a
hydrogen pair at the Gamma point and with two k-points (`tests/unit/test_periodic_ionic.py`); the energy of a short NVE run
is conserved to 1e-3 eV/atom.

## Structural relaxation

`Relaxation/domove.py` is one call of `domove.f90`; `Relaxation/ionic_loop.py`
is the loop of `parsec.F90` around it.

- **`bfgs`**: `Relaxation/lbfgsb.py` is a routine-by-routine translation of
  the L-BFGS-B 3.0 that PARSEC ships (`lbfgsb.f`, `linpack.f`), driven by
  reverse communication with PARSEC's settings (`factr = pgtol = 1e-15`, so
  that the force criterion and not the optimizer ends the run). Its state has
  the layout of the Fortran arrays, which is what makes `relax_restart.dat`
  exchangeable. Its first step moves along the force by `1/|g|`, so the
  second structure often lies higher in energy than the first; the line
  search then corrects the step. This is PARSEC's behaviour.
- **`simple`**: each coordinate moves along its force component by a length
  that grows from `Min_Step` at `Force_Min` and saturates at `Max_Step`.
- **`manual`**: the structures of `manual.dat` are computed one after another
  without recentring and without symmetry.

After a movement everything that depends on the positions is rebuilt on the
unchanged grid (`isolat`, `nonloc`, `corecd`, `ionpot`, `forceion` of
PARSEC): local ionic potential, projectors, core density, ion-ion energy and
the atomic part of the Hartree boundary. The next SCF loop starts from the
previous wavefunctions (the first diagonalization is a subspace-filter step)
and from the previous mixed potential with the ionic part exchanged
(`V_new - V_ion,old + V_ion,new`); the mixer history is restarted. The domain
stays that of the input structure: an atom that leaves it stops the run.

### Start of the next SCF loop

`Potential_Extrapolation` changes where the SCF loop of a moved structure
starts, not where it ends (`SCF/single_point.py`):

- `none`, the default: PARSEC's start above.
- `atomic` (Quantum ESPRESSO's `pot_extrapolation = 'atomic'`): the density
  `rho_out - rho_atoms(R_old) + rho_atoms(R_new)`, the last output density
  with the superposed atomic densities moved along with their nuclei, gives
  the Hartree and exchange-correlation potentials of the first input
  potential, which replace those of the last output potential. The first
  iteration filters with the degree of the first solve (`Chebdav_Degree`),
  so that the saved orbitals catch up with the moved potential before the
  mixer sees their density.
- `second_order` (QE's `'second_order'`): from the second move on, the change
  `d = rho_out - rho_atoms` is extrapolated as well,
  `rho_atoms(R_new) + 2 d(R_old) - d(R_older)`.

The eigensolver always starts from the subspace of the last structure. A
first-order extrapolation of that subspace along an MD trajectory (QE's
`wfc_extrapolation = 'first_order'`: `2 X(t) - X(t-1) R`, with `R` the
polar factor of the overlap of the two bases) was tried and left out: it
saves no SCF iterations, since the density decides when the loop ends, and
only lets the first iteration after a move filter at the later degree. On
C35H36 (`second_order`) that gave 3.2 iterations per step against 3.1 with
the stronger first filter; on C795H300 (four A100) 4.0 against 3.6
iterations, 10.96 against 10.66 s per step and 18 GB more on each GPU,
and C1653H508 no longer fitted.

Without superposed atomic densities as the first density (an imported or
predicted one) and with spin polarization the loop starts as with `none`.
The extrapolation alone saved nothing on nanodiamonds; with the stronger
first filter it saves a quarter to a third of the SCF iterations of a later
structure (numbers under "On A100 GPUs"). Converged tightly, both starts give
Fortran's trajectory; at a loose SCF criterion a trajectory changes within
the SCF noise of its forces, so a relaxation can end one or two structures
earlier or later.

### SCF criterion of a relaxation

With `Relaxation_SCF_Upscale: U` the SCF loop of a structure stops at
`Convergence_Criterion * clamp(F / Force_Min, 1, U)`, `F` the largest force
on a movable atom of the structure before, and that of the input structure
at `U` times the criterion (`adaptive_scf_criterion` in
`Relaxation/ionic_loop.py`; Quantum ESPRESSO's `upscale` scales its
criterion the same way from the other end). A structure whose forces meet
`Force_Min` at a looser criterion is converged again at
`Convergence_Criterion`, continuing from its own loop, before the mover sees
it, so the run ends on forces of the input's accuracy. The log names the
criterion of every structure.

The forces of a looser loop are less accurate. C35H36 at its input
structure (largest force 0.037 Ry/bohr), from superposed atomic densities:

| Criterion | SCF iterations | Energy error | Largest force error |
|---|---:|---:|---:|
| 2e-4 Ry | 13 | 9e-7 Ry | 9e-5 Ry/bohr |
| 5e-4 Ry | 11 | 5e-6 Ry | 5e-5 Ry/bohr |
| 1e-3 Ry | 9 | 5e-5 Ry | 6e-4 Ry/bohr |
| 2e-3 Ry | 7 | 4e-4 Ry | 1.1e-3 Ry/bohr |

The criterion is `U` times looser only where the forces are `U` times
`Force_Min` or more, so the error stays near a hundredth of the force. The
BFGS relaxation of C35H36 (`Force_Min` 0.01 Ry/bohr, `atomic`) took 9
structures and 64 SCF iterations at the fixed criterion, 8 structures and
43 iterations with `U = 10`; the two final structures differ by at most
0.015 bohr, within what `Force_Min` allows. On 16 A100 (inputs of
`gpu_benchmark/`, `atomic`), with `U = 10` against 1:

| Cluster | Structures | SCF per later structure | Program |
|---|---:|---:|---:|
| C795H300 | 7 and 7 | 1.51 s against 1.77 s | 19.2 s against 21.0 s |
| C1653H508 | 6 and 6 | 3.95 s against 4.24 s | 38.0 s against 40.4 s |

The final energies differ by 4e-5 Ry and 2.5e-3 Ry (about 1e-6 Ry per
atom): both runs meet `Force_Min` at slightly different structures.

Symmetry: the point-group operations of the input structure that respect the
movable flags are found once. Forces are averaged over them and so are the
positions the mover returns, which keeps a symmetric structure symmetric to
round-off. When `Move_Flag` fixes an atom that an operation exchanges with a
movable one, the run uses no symmetry and says so.

## Molecular dynamics

`MolecularDynamics/` translates `moldyn.f90`: Beeman steps without friction
(microcanonical) or Langevin dynamics with `Friction_Coefficient`,
the cooling schedules `stair`, `linear` and `log`. The mean force is
subtracted before the integrator and no symmetry is used, as in PARSEC.
`Restart_mode` continues from `mdinit.dat`.

Random numbers decide the initial velocities and the Langevin force, so a
trajectory can be compared with Fortran PARSEC only where both draw the same
ones:

- `test_rng: true` selects PARSEC's own deterministic generator, started from
  `rng_seed`; the numbers are those of the Fortran program.
- `rng_seed: n` with `n > 0` alone makes PARSEC seed the generator of its
  compiler. This package then draws what a PARSEC built by gfortran 8 or
  later draws, so the two give the same trajectory;
  `PARSEC_MD_RANDOM_NUMBERS=numpy` takes NumPy's generator with that seed
  instead.
- Without either, PARSEC seeds from the clock and this package from fresh
  entropy: two runs differ, in either code.

Behaviour of the Fortran that looks unintended is reproduced so that
trajectories can be compared, and is listed in the module documentation of
`MolecularDynamics/moldyn.py`: `Time_Step` is taken in atomic units whatever
unit the input names, masses use 1836 electron masses per atomic mass unit,
and kinetic and potential energies are reported per different atom counts.

## On GPUs and several nodes

Nothing leaves the devices that a single point keeps there.

- **Moved atoms**: the projector matrices of every symmetry sector are
  replaced in place on the device operators, with their replicas on the other
  devices of a shared sector basis. Stencils, potential buffers, recorded
  filter graphs and the sector subspaces stay.
- **Warm start**: each sector solver keeps its subspace and marks it as the
  start of a new SCF loop.
- **Forces**: the nonlocal term is contracted where the orbitals are, sector
  by sector and device by device, and only one small array per projector is
  downloaded. The local term runs in a CUDA kernel on the devices of the
  process that holds the density (in a run on several nodes, the four GPUs of
  the root node), each of which sums a part of the atoms. Where symmetry
  sectors are solved the sums run over the symmetry wedge of the grid and the
  group operations complete them, which costs as many times less as the group
  has operations and changes the forces by round-off only;
  `PARSEC_LOCAL_FORCE_SUMS=full` sums over the whole grid as before.
- **Projector gradients**: once the atoms of a structure are in place, a host
  thread samples the gradients of its projectors and reduces them to the
  sectors of the process while the devices solve its SCF loop; the forces
  then take them (`PARSEC_CUPY_FORCE_PREFETCH=on`, the default; `off` builds
  them when the forces are asked for). Nothing reaches a device before the
  forces, and the forces are the same to the last bit.
- **Several nodes**: the rank of each node rebuilds the projectors of its own
  sectors (command `geometry`), restarts its solvers (`new_scf_loop`) and
  contracts its part of the nonlocal force (`forces`); the root sums them.

The multi-node runner divides the symmetry sectors of the structure among its
ranks, one per node. A structure without point-group symmetry, a relaxation
whose movable flags break the symmetry, a `manual` run and molecular
dynamics have one sector, the identity. Launched with one rank, the runner
solves it on the GPUs of one node, which share its basis column by column
(`PARSEC_CUPY_SINGLE_SECTOR=auto`, the default; `on` takes this route outside
the runner too, `off` never). The numbers are those of the full grid; each
GPU holds a quarter of the orbitals. `--serial-control` keeps the former
route, on which the first GPU holds them all.

On several nodes such a run starts one rank per GPU
(`scripts/run_multi_gpu_scf.sh --nodes N --rank-per-gpu`), and the ranks
share the basis of the one sector (`acceleration/Eigensolvers/mpi_shared_basis.py`):
each holds all grid rows of its columns with a replica of the operator and
filters them alone; the Ritz step moves the filtered columns and `H` times
them to row blocks (`Alltoallv` of GPU arrays), sums the Gram matrices,
lets the root solve the small problem and broadcast the coefficients, and
moves the rotated rows back. Densities and nonlocal forces are summed over
the ranks; the root alone runs the Hartree solve, the mixer and the mover.
The trial basis is the single-device DLARNV stream, so the runs give the
trajectory of one node. Molecular dynamics with `second_order` on A100,
per later structure (`md_nrg.dat` equal to that of one node to the last
printed digit in every case, 3.6 SCF iterations per structure):

| Cluster | One node, one rank | 4 ranks, one node | 8 ranks, two nodes | 16 ranks, four nodes |
|---|---:|---:|---:|---:|
| C795H300 | 10.66 s, 14.2 GB | 13.07 s | 9.65 s | 6.77 s, 15.6 GB |
| C1653H508 | 40.47 s, 39.8 GB | | | 23.02 s, 39.0 GB |

On one node the shared basis of a single rank is faster: its devices copy
their blocks to each other directly. Across nodes the exchanges carry about
17 GB/s per rank, near what the network interface of a GPU carries, and take
half of the eigensolver's time on 16 ranks (C795H300: per Ritz step 0.37 s to
rows, 0.19 s back, against 0.32 s of filtering). A filter pipelined with
those exchanges was measured slower and left out (see the module).

## Differences from Fortran PARSEC

| Subject | Fortran PARSEC | This package | Why |
|---|---|---|---|
| Forces of a single point | Always computed | Only with `Skip_Force: false` | Existing inputs keep their cost. |
| Periodic forces | Reciprocal-space form factors (`force_loc_pb.f90`) | Real-space sums over images, with an FFT version for the potential | The same terms; the grid-sampled potential has egg-box effects the form-factor one has not (see "Periodic cells"). |
| Periodic forces and symmetry | Symmetrized with the space group | Not symmetrized | No space-group operations are found. |
| Positions of a periodic run | Wrapped into the cell (`back_to_cell`) | Continuous; the builders fold them | The optimizer needs continuous coordinates. |
| SCF-correction force with `Cubic_Spline` | Uses the derivative table together with the spline coefficients of the density | Interpolates the atomic valence density linearly in that term | The combination in `forcloc.f90` mixes two tables; the term vanishes at SCF convergence. |
| `atom.cor` of a restarted run | "Step # 0" shows the coordinates of the input file | Shows the restart structure, which is the one computed | The file then describes the run. |
| Movable flags that break the symmetry | Symmetrizes regardless | Runs without symmetry | The relaxation would leave the group it started in. |
| Positions after a movement | As the optimizer returns them | Averaged over the symmetry operations | Removes round-off drift; no change at print precision. |
| Energy that is not finite | L-BFGS-B reports a converged minimum | The run stops with an error | The Fortran outcome is wrong. |
| Hartree boundary of larger clusters | Multipoles at `Solver_Lpole` | Higher order and atomic tail by default | See "Physical scope" in the package README; `Hartree_Boundary_Tolerance: off` restores PARSEC's boundary for a comparison. |
| `Solver_Lpole` | Read and printed, then order 9 is used whatever it says (runs with 2, 4 and 9 give identical results) | The given order is the lowest one used | Not specific to moving atoms; give 9 for a comparison. |
| PBE | PARSEC's discretization | A discrete-variational one | Energies of the two codes differ by about 1e-4 Ry, so PBE forces have no Fortran reference; they are checked by finite differences. |
| Restart of molecular dynamics | gfortran builds cannot read their own `mdinit.dat` | Reads it | A format statement in `initial.f90`. |

One difference is not a choice. Where the optimizer stalls, that is where a
line search tries steps whose energy differences are below the SCF noise, the
two codes take different decisions from numbers that agree to all printed
digits, and their trajectories part. Builds of the Fortran with different
BLAS libraries differ among themselves in the same way.

## Validation

Every comparison with Fortran PARSEC uses the same input for both codes and
adds `Hartree_Boundary_Tolerance: off` on the Python side, so that both use
PARSEC's Hartree boundary. The Fortran is PARSEC built with gfortran 13.3 and
run on 4 to 12 MPI ranks.

### Forces at a fixed structure

Both SCF loops converged to 1e-8 Ry. Every force table PARSEC prints (ion-ion,
local, SCF correction, core correction, nonlocal, total) agrees within its six
printed decimals, and the total energies within 3e-8 Ry.

| Structure | What it exercises |
|---|---|
| H2O, asymmetric | LDA, s projector |
| H2O with `Cubic_Spline` | spline interpolation of potential and projectors |
| Ge2H2 | nonlinear core correction, s and d projectors |
| Ag2H2 | p and d projectors, relativistic pseudopotential file |
| SiNa4, also displaced and as a cation | old Martins format, occupations from `occup.in` |

`tests/unit/test_forces_parsec_reference.py`,
`tests/unit/test_sina4_parsec_reference.py` and
`acceleration/tests/test_forces_geometry.py` hold these comparisons. PBE forces
are compared with finite differences of the energy instead.

### Trajectories

"Identical" means: the same number of structures, and every energy per atom
and largest force of `atom.cor` equal in all printed digits.

| Case | Structures | Result |
|---|---:|---|
| PARSEC example `0d_water_geoopt`, as shipped (SCF criterion 2e-4 Ry) | 7 | energies within 3e-5 eV/atom, forces within 8e-5 Ry/bohr |
| the same with SCF criterion 1e-8 Ry | 7 | energies identical, forces within 1e-5 Ry/bohr, coordinates within 1e-6 bohr |
| PARSEC test `tests/relaxation` (SiNa4), as shipped | 4 | identical, coordinates within 3e-6 bohr |
| the same with SCF criterion 1e-8 Ry | 4 | identical, coordinates identical |
| water, BFGS, symmetric and without symmetry | 8 and 17 | identical, `iterate.dat` identical |
| water, steepest descent | 7 | identical |
| water, `manual.dat` | 3 | identical |
| water, stopped by `Movement_Num` and continued with `Relax_Restart` | 3 + 14 | identical |
| the same, Fortran continuing the file written by Python | 14 | identical to Fortran continuing its own |
| nanodiamond C35H36, `Cubic_Spline`, `Read_VCD`, SCF criterion 1e-7 Ry | 9 | energies identical, forces within 1e-5 Ry/bohr, coordinates within 2.1e-5 bohr |

Molecular dynamics, both SCF loops converged to 1e-8 Ry: `md_nrg.dat` is the
same text as the Fortran file in every case below, and positions, velocities
and accelerations of `md_mech.dat` agree within one unit of their last
printed digit.

| Case | Steps |
|---|---:|
| water, microcanonical | 8 |
| water, Langevin at 300 K, `test_rng` | 6 |
| water, Langevin with `stair`, `linear` and `log` cooling | 5 to 6 |
| water, Langevin with `rng_seed` and the generator of gfortran | 6 |
| water, microcanonical with a fixed atom (`Move_Flag`) | 5 |
| water, restarted at step 5 from an `mdinit.dat` written by Fortran | 3 |
| PARSEC example `0d_silver_cluster_md` (Ag8) with `test_rng` | 5 |

The silver example as shipped seeds from the clock, so only its first
structure can be compared with the reference output of the example: the
energies agree in all printed digits but the last. At the SCF criterion of
the example (2e-4 Ry) the two codes differ after five steps by 8e-4 eV/atom
in the kinetic energy, which is how far each is from its own tightly
converged trajectory. One structure of Ag8 takes 31 s with Fortran PARSEC on
4 ranks and 7 s with this package on a laptop GPU.

Two water cases, bounded BFGS (`Max_Step` given) and fixed atoms
(`Move_Flag`), agree for 20 and 31 structures; there the line search works on
energy differences at round-off and the two optimizers decide differently, as
described under
"Differences from Fortran PARSEC".

The readable path and the accelerated path give the same trajectories. The
nanodiamond relaxation takes 325 s with Fortran PARSEC on 12 ranks and 60 s
with this package on a laptop GPU (RTX 5070, 8 GB).

### On A100 GPUs, one to four nodes

Perlmutter, four A100 80 GB per node, double precision, no cache.

Equality of the execution layouts:

| System | Compared | Result |
|---|---|---|
| water, BFGS | one-rank serial control, 1, 2 and 4 ranks against the Fortran trajectory | identical |
| SiNa4, BFGS | the same four layouts against the Fortran trajectory | identical |
| C35H36, forces | serial control against 1, 2 and 4 ranks, with sector bases shared by 2 and 4 GPUs | largest difference 9e-14 Ry/bohr |
| C87H76, BFGS | serial control, 1, 2 and 4 ranks | identical, 6 structures |
| C795H300, forces | 4, 8 and 16 GPUs | largest difference 7e-13 Ry/bohr |
| C795H300, BFGS | 4, 8 and 16 GPUs | identical, 7 structures |
| water and Ag8, molecular dynamics | one node against the Fortran `md_nrg.dat` | same text |

Relaxation of C795H300 (1095 atoms, 3480 valence electrons, 0.2 Angstrom
grid, CA-LDA) from its GFN2-xTB structure, with the solver controls of the
measured single points (`Eigensolver: chebff`, `FF_MaxIter: 4`,
`Chebdav_Degree: 30`, `Chebyshev_Degree: 15`):

| Run | GPUs | Structures | Program time | Later structure: SCF, forces, rebuilding |
|---|---:|---:|---:|---|
| `Force_Min` 0.01 Ry/bohr, SCF criterion 2e-4 Ry | 4 | 7 | 47 s | 4.5 s, 0.6 s, 0.26 s |
| the same | 8 | 7 | 35 s | 3.0 s, 0.6 s, 0.26 s |
| the same | 16 | 7 | 30 s | 2.3 s, 0.5 s, 0.26 s |
| `Force_Min` 0.001 Ry/bohr, SCF criterion 1e-6 Ry | 16 | 17 | 85 s | 3.6 s, 0.5 s, 0.27 s |

The first structure costs about 6 s on 16 GPUs, as a single point does.
The energy falls from -9437.8200 to -9438.0419 Ry at the first criterion and
to -9438.0569 Ry at the second. A run stopped after three movements and
continued from its `relax_restart.dat` reaches the criterion in the same
number of structures; its energies differ by up to 5e-5 Ry from the uninterrupted
run, because the continued run starts its SCF loop without wavefunctions and
the SCF criterion is 2e-4 Ry.

For the next size, C2455H636 (3091 atoms) on 16 GPUs, a later structure costs
11.3 s of SCF, 1.7 s of forces and 0.8 s of rebuilding. Of the 1.7 s, 1.0 s
are the projector gradients, which every node builds on its host, 0.25 s the
local sums and 0.2 s the ion-ion forces. For C795H300 the three parts take
0.35, 0.07 and 0.04 s. The local sums grow with the number of atoms times the
number of grid points: for C2455H636 they took about 2.8 s over the whole grid
on one GPU and 1.8 s on the four GPUs of the root node, and take 0.25 s on
those four GPUs over the wedge of the D2 group.

### Starts of the SCF loop and molecular dynamics on A100 GPUs

The nanodiamonds of `examples/0d_nanodiamonds`. Fortran PARSEC is the same
version built on Perlmutter's CPU nodes (GNU compilers, 128 MPI ranks per
node); both codes use the inputs of `fortran_comparison/` with the SCF
criterion lowered to 1e-7 Ry. BFGS:

| Cluster | Structures | `none` | `atomic` | `second_order` |
|---|---:|---|---|---|
| C35H36 | 9 | coordinates within 1.6e-5 bohr | within 5.9e-5 bohr | |
| C87H76 | 6 | within 3e-6 bohr | within 1.4e-5 bohr | |
| C185H124 | 7 | within 3.6e-5 bohr | within 1.4e-5 bohr | within 2.1e-5 bohr |

Energies per atom and largest forces agree within one unit of their last
printed digit, with the same number of structures. Molecular dynamics at
300 K (5 steps of 20 a.u., `test_rng`): `md_nrg.dat` of C35H36 and C87H76
(the latter also with `second_order`) agrees with the Fortran file within
one unit of the last digit. Fortran PARSEC needed the 3500 s of its job for
five rows of C185H124 on one node; those five agree, and this package took
128 s on four GPUs.

Relaxations with the inputs of `gpu_benchmark/` (`Force_Min` 0.01 Ry/bohr,
SCF criterion 2e-4 Ry) on 16 GPUs. A later structure of C795H300:

| Start | Gradients | SCF | Forces |
|---|---|---:|---:|
| `none` | built for the forces | 2.20 s | 0.52 s |
| `atomic` | built for the forces | 1.71 s | 0.53 s |
| `atomic` | built beside the SCF loop (default) | 1.73 s | 0.14 s |

Program times, gradients beside the SCF loop:

| Cluster | Structures | `none` | `atomic` |
|---|---:|---:|---:|
| C1213H412 | 4 | 27.3 s | 25.7 s |
| C1653H508 | 6 | 54.2 s | 46.0 s |

At this SCF criterion the forces carry noise of about 1e-3 Ry/bohr, so the
two starts can end a relaxation at different structures: with the inputs
of `fortran_comparison/` C795H300 took 7 structures with `atomic` and 5 with
`none`. `second_order` saves nothing in BFGS, whose steps do not follow a
smooth path (C795H300 on 4 GPUs: 4.13 s of SCF per later structure against
3.44 s with `atomic`).

Molecular dynamics (5 steps) on the four GPUs of one node, per later
structure and largest memory of one GPU; C185H124 to C795H300 with the
first-solve controls of the GPU benchmark, the larger clusters with the
inputs of `gpu_benchmark/`:

| Cluster | `--serial-control` | Sector of the identity, `none` | `atomic` | `second_order` |
|---|---|---|---:|---:|
| C185H124 | 4.33 s, 17.1 GB | 2.99 s, 4.2 GB | 2.73 s | 2.02 s |
| C459H204 | 17.6 s, 69.4 GB | 9.8 s, 9.1 GB | 8.1 s | 5.9 s |
| C795H300 | out of memory (inputs of `fortran_comparison/`) | 18.1 s, 14.2 GB | 14.2 s | 10.7 s |
| C1213H412 | | 41.9 s, 27.0 GB | 32.7 s | 25.5 s |
| C1653H508 | | 69.0 s, 39.8 GB | 54.2 s | 40.5 s |

A later structure takes 6 to 7 SCF iterations with `none`, 5 with `atomic`
and 3 to 4 with `second_order`.
