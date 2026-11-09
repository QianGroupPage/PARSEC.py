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
- Self-consistent hybrid functionals of isolated systems (PBE0, HSE06,
  B3LYP, HF, ...), without or with spin polarization: forces, relaxation and
  molecular dynamics on both command lines (`main.py` on the GPUs of one
  node) and on several nodes (the MPI runner of
  [acceleration/MULTI_GPU.md](acceleration/MULTI_GPU.md#hybrid-functionals-on-several-nodes));
  of periodic cells (bulk at the Gamma point, also with spin, and with
  k-points; slabs) on the readable solver and on the GPUs of one node; see
  [Hybrid functionals](#hybrid-functionals) below.
- Periodic cells (`Boundary_Conditions: bulk`), without spin-orbit coupling, at the Gamma point or with a Monkhorst-Pack
  k-point sampling (collinear spin polarization at the Gamma point only): forces, `Minimization: bfgs | simple | manual`
  and molecular dynamics. See "Periodic cells" below; the spin-unpolarized Gamma-point cell also runs on the accelerated
  path, and a cell with k-points on the GPUs of the k-point solver (see "On GPUs"). A spin-polarized one with k-points
  does not have them yet.
- Self-consistent spin-orbit coupling (`SO_from_scratch`, alone or with `Spin_Polarization`) and non-collinear magnetism
  (`Non_Collinear_magnetism`, with or without spin-orbit), isolated or periodic at the Gamma point or with k-points:
  forces, relaxation and molecular dynamics. See "Spin-orbit coupling and non-collinear magnetism" below. The
  perturbative spin-orbit correction (`SO_PSP` without `SO_from_scratch`) has none.
- With `Use_Fortran_Routines: .true.` (and for the inputs that run on those routines by default - slabs, shifted bulk grids, forces of a bulk cell at the
  Gamma point on a grid both sets share with Cubic_Spline species, bulk cells whose space group sets Fortran's grid -
  or by themselves, see
  `Input/routines.py`), the Fortran-derived routines:
  - Periodic (bulk) cells with a fixed cell: forces, relaxation and molecular
    dynamics at Gamma and with k-points (also with collinear spin, with an
    irreducible k-point wedge and with `Double_Grid_Order > 1`), on the
    orthorhombic and on Fortran's general grid; see
    [Periodic cells](#periodic-cells).
  - Wires and slabs (`Boundary_Conditions: wire` / `slab`): forces (Gamma,
    collinear spin, k-points with any weights, spin with k-points), relaxation
    and molecular dynamics; see
    [Wires and slabs](#wires-and-slabs). Spin-orbit coupling, non-collinear
    magnetism and `Double_Grid_Order > 1` are refused for wires and slabs.
  - Spin-orbit coupling (`SO_from_scratch`/`SCF_SO`, also with collinear spin)
    and non-collinear magnetism, isolated or periodic (bulk); see
    [Spin-orbit coupling and non-collinear magnetism](#spin-orbit-coupling-and-non-collinear-magnetism).
    Refused: forces with the perturbative spin-orbit correction (`SO_PSP`
    without `SO_from_scratch`).
- A spin-polarized calculation uses the same routines: `forcloc.f90` reads
  the total density and the averages over the two channels of the potential
  difference and of the exchange-correlation potential, and `forcnloc.f90`
  sums over the states of both channels with the weight `4/nspin`.
- Not translated from `forcloc.f90` and `forcnloc.f90`: point charges,
  LDA+U and van der Waals corrections.
- With `Use_Fortran_Routines: .true.` (and for the inputs that run on those routines by default - slabs, shifted bulk grids, forces of a bulk cell at the
  Gamma point on a grid both sets share with Cubic_Spline species, bulk cells whose space group sets Fortran's grid -
  or by themselves, see
  `Input/routines.py`), the Fortran-derived routines:
  - The spin-orbit term of `forcnloc.f90` (`forcso`) is in
    `Forces/spin_orbit.py` (section below); the point-charge, LDA+U and TS
    terms are those of `Forces/total.py::correction_forces` and
    `Forces/ion_ion.py`.
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
| `Write_Relax_Restart` | `true` here | PARSEC 1.5: write `relax_restart.dat` after every movement. 1.5 writes it only with `true` (its default is `false`); this package writes it unless the input says `false`. |
| `Calculate_Force` | not given | PARSEC 1.5: `true` gives the forces of a single point (as `Skip_Force: false`), `false` none. Without either keyword a single point computes no forces here (PARSEC computes them in every run). `false` with a relaxation or molecular dynamics is refused. `Use_Old_Forces` (1.5's serial force routines, the same forces) is accepted and changes nothing. |
| `Recenter_Atoms_In_Aperiodic_Directions` | `true` | PARSEC 1.5: `false` keeps the coordinates of the input (no re-centring of a cluster or, along z, of a slab; also in a molecular dynamics restart). Since 1.5 a slab is re-centred in a `Restart_Run` too. |
| `Move_Flag` (per `Atom_Type`) | 0 | 0 moves every atom of the type, `n > 0` its first `n` atoms, a negative value reads a fourth column of `Atom_Coord` (1 moves, 0 fixes). |
| `Molecular_Dynamics` | `false` | Switches molecular dynamics on; it excludes `Minimization`. |
| `Cooling_Method`, `Tinit`, `T_Final`, `T_Step`, `Time_Step`, `Friction_Coefficient`, `Step_num`, `Restart_mode`, `rng_seed` (PARSEC 1.5: `Mol_RNG_Seed`), `test_rng` | PARSEC's | Controls of `moldyn.f90`; see "Molecular dynamics" below. |
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
stopped above `Force_Min`).  As in PARSEC 1.5, an SCF loop that does not
converge ends a relaxation or molecular dynamics run: the forces of that
structure are computed and reported, the atoms are not moved from it, and the
log says " WARNING !!! atom movement is terminated" (the previous Fortran tree,
and this package before, moved on).

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
`atom.cor` and `movie.xyz`, in the formats of PARSEC 1.5: integer fields `i6`
for the steps and `i7` for the atom count, and the velocities of
`md_mech.dat` (and so of an `mdinit.dat` cut from it) are `v(t)` of the step,
where the previous Fortran tree wrote its estimate of the next step's velocity.

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

### Hybrid functionals

With a hybrid `Correlation_Type` ([README.md](README.md#self-consistent-hybrid-functionals))
the forces keep the form above, with the orbitals and the density of the
converged generalized Kohn-Sham state: with norm-conserving
pseudopotentials the exchange operator has no explicit dependence on the
ions, and the only term that sees the core density is the semilocal part,
whose potential (with the reduced semilocal exchange) the core-correction
force reads. Finite differences of the total energy (H2, H2O and LiH, whose
Li has a core correction, `h = 0.4` bohr, displacements of 0.01 bohr) give
the same analytic-minus-numeric difference for PBE0 as for PBE, 1.2e-3 to
3.4e-3 Ry/bohr (the grid error of that spacing), equal within 1e-4 Ry/bohr.
In a relaxation or in molecular dynamics the Fock loop of a moved structure
builds its first exchange operator from the orbitals of the structure
before, so only the first structure runs the PBE start.

On several nodes (`scripts/run_multi_gpu_scf.sh`) the ranks keep the orbitals
of their sectors between the structures as well, and the nonlocal force terms
come from the sectors of every rank as for a semilocal functional; a
structure without symmetry (molecular dynamics, `manual` moves) has its sector
on one GPU of the root and the pair densities of its exchange on the GPUs of
all ranks. Measured on A100 nodes: two BFGS moves of C185H124 (PBE0, forces
of every structure) on 2 nodes against one node, energies within 6.7e-10 Ry,
forces within 1.2e-11 Ry/bohr, the same SCF iterations; three steps of
molecular dynamics of H2O (PBE0, one sector) on 2 nodes, the trajectory of one
node to the last bit.

Periodic cells ([README.md](README.md#periodic-cells)) take the same forces:
at the Gamma point from the real orbitals of the Fock loop (also with
collinear spin), with k-points from the Bloch factors of every k-point the
loop solved (the time-reversal representatives of the mesh, whose weights
count their partners); on Andrew Grieder's routines the nonlocal force is
accumulated from them after the loop
(`SCF/hybrid_periodic.bloch_nonlocal_forces`), on the Fortran-derived ones
`Forces/periodic.py` reads them. A moved structure starts its Fock loop
from the potentials, the eigensubspaces of every k-point and the Bloch
orbitals of the structure before. There is no stress.

### Meta-GGA functionals

With a meta-GGA ([README.md](README.md#meta-gga-functionals)) the forces keep
the form above as well: `tau` is a functional of the orbitals alone, and the
orbital term `-div(vtau grad)` (with the first-order term of `W = de/dg` in
the orbital form of the density gradient) has no explicit dependence on the
ions. The only term that sees the frozen core density is the core
correction. The core density enters the density `n` and the density
gradient, never `tau` (as in Quantum ESPRESSO and PARSEC 1.5 with libxc), so
the force reads `dE_xc/dn_core`:

- density form: the exchange-correlation potential `vrho + D^T (2 vsigma D n)`
  of the total density, as for a GGA;
- orbital form: `vrho + D^T W` with `W` formed from the gradient of the valence
  orbitals plus that of the core density
  (`MetaGGARecord.core_potential`, half the core density per channel with
  spin), since the potential of that form holds `vrho` alone.

The SCF test of `tests/unit/test_meta_gga_scf.py` compares this force with
the derivative of the meta-GGA energy with respect to the position of the
core density (LiH, both forms): 2e-4 relative, as for PBE. The SCF
correction force uses `V_out - V_in` as for every functional; the residual
of `vtau` (and `W`) is not part of it. In a relaxation or in molecular
dynamics the next structure starts from the mixed `vtau` (and `W`) of the
structure before, so only the first one runs the PBE start.

Periodic cells (bulk at Gamma and with k-points, slabs and wires) take the same terms: the periodic and slab
forces read `dE_xc/dn_core` through `Forces.periodic.meta_gga_core_potential` (the core potential of the orbital form,
the exchange-correlation potential otherwise), and the k-point loops carry `vtau` and `W` to the next structure
(Andrew Grieder's backend through `SCFWarmStart`, the Fortran-derived one through `meta_gga_start` of
`run_scf_kpoints`). Central differences of the total energy of a Si cell (fcc, two atoms, one displaced by
(0.1, 0.05, -0.08) bohr, Andrew Grieder's routines, step 0.01 bohr, SCF to 1e-9 Ry) against the forces of
`compute_periodic_forces`, largest component difference in Ry/bohr (forces ~0.03-0.06):

| functional | Gamma, h 0.4 | 2 k-points, h 0.4 | Gamma, h 0.3 |
|---|---|---|---|
| PBE | 3.5e-4 | 2.7e-4 | 1.7e-4 |
| smooth test meta-GGA (`tests/unit/test_meta_gga.py`) | 3.4e-4 | - | - |
| SCAN, density form | 1.8e-3 | 2.5e-3 | 3.8e-4 |
| SCAN, orbital form | - | 2.6e-3 | - |

The meta-GGA terms of the forces are complete: the smooth meta-GGA matches PBE, and SCAN's larger difference at
h 0.4 falls by a factor of five at h 0.3 (PBE's by two), the grid sensitivity of SCAN rather than a missing term. On
the Fortran-derived routines (`Use_Fortran_Routines: .true.`, the same cell with the FHI98PP potential of
`examples/3d_meta_gga`, two k-points, h 0.3, `Forces.compute_forces`) the differences are 2.1e-3 (PBE) and 2.6e-3
Ry/bohr (SCAN): the meta-GGA adds nothing to what those routines show for PBE.

## Periodic cells

A periodic cell has forces, relaxation and molecular dynamics through the periodic backend
(`Relaxation.periodic_backend`) of the readable solver. `main.py` hands such an input to `parsec_python.cli`
(`NOTE: ... running the reference parsec_python.cli solver instead`) when it has k-points or spin; a spin-unpolarized
Gamma-point cell stays on the accelerated path.

Forces (`Forces.compute_periodic_forces`; the Fortran routines are `forpbc.f90`, `force_loc_pb.f90`, `ewald_sum.f90` and
`forcnloc.f90`):

1. **Ion-ion**: the exact derivative of the Ewald energy (`Forces/ion_ion.py: ewald_ion_ion_forces`): the real-space sum
   over the atom pairs within the cutoff and the reciprocal sum over the structure factors of a box of reciprocal
   vectors (`V_ion.ewald.ewald_ion_ion_terms`, 4 s for 4,096 atoms on a laptop).
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

**Spin polarization (Gamma point).** A spin-polarized periodic cell that asks for forces, a relaxation or molecular
dynamics runs the loop of the isolated spin-polarized path (`run_spin_scf`, which `run_scf` selects; the periodic system
supplies its starting spin densities, spin-polarized XC and the `alpha` energy). The forces are the unpolarized ones on the
total density, the average of the two channels' exchange-correlation potential and the orbitals of both channels at one
electron per state; there is no term for the SCF residual. A spin-polarized input with k-points, spin-orbit coupling or
non-collinear magnetism has no forces. Checked three ways (`tests/unit/test_periodic_spin_forces.py`): a non-magnetic spin run
has the energy and forces of the unpolarized run (2e-6); each term is the derivative of the spin energy functional at the
converged orbitals (ion-ion 1e-12, local ionic 1e-7, nonlocal 1e-8 Ry/bohr with the cubic spline; the core term as in the
unpolarized case); and against Fortran with the same fixed occupations (`occup.in`, 6 up and 5 down states, three Ge atoms
with one electron removed): total force 2.0e-3 / 6.3e-4 at 24 / 32 points (2.0e-3 / 6.1e-4 unpolarized), nonlocal
3.2e-5 / 5.2e-6. Two Fortran conventions show in such a comparison. Its `alpha` energy is multiplied by the total ionic
charge, not by the number of electrons, so a charged cell differs by a constant (0.191 Ry per electron of charge for this
cell; forces are not affected). And `rho_har.F90` shifts the exchange-correlation potential of each spin channel to zero
average in a periodic cell, which removes the average exchange splitting between the channels: with Fermi smearing, where
both channels share one Fermi level, Fortran settles at a smaller moment (0.69 against 0.94 Bohr magnetons for that cell at
1000 K) than the LSDA solution that PARSEC.py keeps; with fixed occupations the two agree.

## Spin-orbit coupling and non-collinear magnetism

`SO_from_scratch` and `Non_Collinear_magnetism` solve for two-component spinor states, each holding at most one electron.
Their forces are the Hellmann-Feynman forces of the energy the SCF loop minimized, in the pieces of the spin-polarized
case:

- **Ion-ion, local ionic, core correction, SCF residual**: the total density, the exchange-correlation potential averaged
  over the two spin directions (`V_avg` of the local-rotation form, whose `B_xc.sigma` part does not depend on the
  positions), and the last `V_out - V_in` of the averaged potential. The result records these (`xc_potential`,
  `input_effective_potential`, `output_effective_potential`).
- **Kleinman-Bylander projectors**: they act alike on both spin components, so the force is that of the spin-polarized
  case on the `2N` components at the weight `f_n/2` (`Forces/spin_orbit.py`, `spinor_nonlocal_forces`).
- **Spin-orbit projectors**: the spin-orbit term is a bilinear form in the overlaps `<P_j|u_sigma>` of the spinor with the 16
  projector functions of an atom (`V_ion` and `V_so` for the harmonics of `l = 1, 2`, complex spherical harmonics, KB sign
  per `l`); `apply_spin_orbit` multiplies the projectors with coefficients linear in these overlaps (`lzsz` on the same
  spin, the `lsxy` ladders on the other). A projector depends on the atom only through `r - R_a`, so
  `F_a = 2 sum_n f_n Re sum_(sigma,j) conj(<grad P_j|u_n,sigma>) c_(sigma,j)`. The radial functions are interpolated
  linearly in the Hamiltonian, so the gradient takes the slope of that interpolant: the force is the derivative of the
  energy that is minimized, not of a smoother table (Fortran's `dv_ion`/`dv_so` tables).

Isolated runs use `compute_spinor_forces`; a periodic run uses `compute_periodic_forces` on a result whose nonlocal force
the k-point job accumulated from the orbitals of every k-point in a last pass (`compute_forces=True`, as for the scalar
k-point driver), with the Bloch phases of the projectors and their gradients and the atoms folded into the cell. Moved
atoms keep the direction of their initial moment (`Initial_NCL_Moment`). The SCF loop of a moved structure continues from the
previous one, for plain spin-orbit coupling, collinear spin and non-collinear magnetism alike (isolated or periodic, Gamma
point or k-points): the last mixed potential with the new ionic potential (for spin the pair `V_up`, `V_down`; for
non-collinear magnetism `V` and the exchange-correlation field `B_xc`, which carries the moments to the moved atoms, so
`Initial_NCL_Moment` is not asked for again), the Hartree potential as the start of the Poisson solve, and the eigensubspace
of every k-point (kept on the host between structures; on the GPUs it goes back to the device when the k-point is solved),
so the first iteration filters the old orbitals instead of solving from random vectors. `Potential_Extrapolation` applies
to plain spin-orbit coupling only (the atomic density guess is a total density, as for the scalar spin-polarized loop). On
Au2 in a 9 bohr cell at 0.45 bohr spacing (Gamma point, CPU) a molecular dynamics step took 7 to 9 SCF iterations instead of
21 to 31 (plain spin-orbit 9 against 31, collinear spin 7 against 24, non-collinear 7 against 23), the wall time of a
relaxation 111 s instead of 195 s (12 structures, the same path). The gain shrinks as the steps grow: with steps of a few
tenths of a bohr (a relaxation on a 0.7 bohr grid, whose forces are several Ry/bohr) it vanishes. The direction of a
non-collinear moment is a soft mode, so a continued and a fresh loop agree on its size and to a few hundredths of a
Bohr magneton on its components. The forces are not symmetrized over the point group of
the atoms for a spin-polarized or non-collinear run, whose magnetization need not respect it; the net force is removed.

Fortran's `forcnloc.f90` has a spin-orbit term (`forcso`, `soforc.F90`) but warns that it "may be inaccurate", and its
`flsxy` pairs the same harmonic index on both sides where the Hamiltonian pairs `lm` with `lm +- 1`; with the divisor
`nspin/mxwd` of `forcnloc.f90` for a spinor it is not a reference. These forces are validated against the energy of this
package instead (`tests/unit/test_spinor_forces.py`, Au dimer with a relativistic pseudopotential, cubic spline):

| Check | Result |
|---|---|
| Each term against a finite difference of its energy at the converged spinors (non-collinear, spin-orbit, one unpaired electron) | ion-ion 1e-8, local ionic 1e-6, nonlocal plus spin-orbit 7e-5 Ry/bohr |
| Spin-orbit term alone, smooth orbitals, step 1e-4 | 1e-5 Ry/bohr; the residue is the kinks of the piecewise-linear radial tables |
| Total force against the derivative of the converged SCF energy (Richardson, steps 0.01 / 0.005 bohr), isolated, non-collinear with spin-orbit | -9.5720 against -9.5725, 4.9827 against 4.9817 Ry/bohr |
| The same, collinear spin with spin-orbit (isolated and 2x1x1 k-points), unpolarized spin-orbit (Gamma), non-collinear (2x1x1 k-points, periodic) | 5e-5 to 3e-3 Ry/bohr on forces of 0.7 to 10 Ry/bohr |
| Without spin-orbit, non-collinear against the collinear spin-polarized driver (whose forces were compared with Fortran) | energy 1e-6 Ry, forces 2e-4 Ry/bohr |

The forces on this dimer are large (10 Ry/bohr, with a net force of the same size before it is removed) because a
0.7 bohr grid is far too coarse for Au; they show that the forces follow the energy of the grid, which is what the checks
test, not that the grid is converged.

Not covered: the perturbative correction (`SO_PSP` with an ordinary spin-polarized SCF), which shifts the eigenvalues of that
SCF to first order and reports no total energy that includes the shift, so a relaxation would have no energy to follow
(`SO_from_scratch` has forces); `Energy_Refinement`; the row-distributed MPI k-point solver (it raises, as for the scalar k-point forces); the
accelerated path, which hands every spin-orbit and non-collinear input with forces to the reference solver (the pooled
k-point solver of that solver runs on the GPUs for periodic and non-collinear inputs and forms the overlaps of the force on the
device, which agrees with the CPU to 1e-5 Ry/bohr on four A100 GPUs: `parsec_tests/pbc_forces/gpu_spinor` and `gpu_k`).

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
  `rng_seed`; the numbers are those of the Fortran program.  A restart draws
  `2 (iframe - 1)` numbers first, as PARSEC 1.5 does, so that the generator
  continues about where the earlier run left it.
- `Mol_RNG_Seed` is PARSEC 1.5's name of `rng_seed` (1.5 no longer reads the
  old name); both are read here with the same meaning, an input with both is
  refused.
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

## Periodic cells

`Forces/periodic.py` evaluates the forces of a bulk cell with the structure of
Fortran's `forpbc.f90`/`force_loc_pb.f90` (local and core-correction terms),
`ewald_sum.f90` (ion-ion) and `forcnloc.f90` (projectors with their lattice
replicas and, for k-points, the Bloch phases `right`), then symmetrizes them
and removes the net force. Each term is the exact derivative of the energy this
package computes, which differs from Fortran's in its local potential (a
real-space Ewald split instead of the G-space `v_first`/`pot_local`):

- local: `dV sum_r rho(r) [sum_T f'(|d_T|) d_T/|d_T| + 2Z (4 pi/V) sum_G w_G G sin(G.(r - R))]`
  with `f = V_loc + 2Z erf(eta r)/r` and the translations and G vectors of
  `ewald_local_ionic_potential`; ion-ion: the derivative of
  `ewald_ion_ion_energy` over the same atom pairs within the real-space cutoff
  and the same reciprocal vectors (`V_ion.ewald.ewald_ion_ion_terms` evaluates
  both; both sums keep every image a pair or a grid point reaches, for any cell
  shape, so an atom that crosses a cell face and is translated back changes no
  energy);
- radial functions are differentiated as the energy samples them (slope of the
  linear interpolant of `r V`, of the densities and of the projectors, or the
  spline derivative). `PARSEC_PERIODIC_FORCE_SLOPES=tabulated` takes for the
  projectors Fortran's `dvw` table instead, which reproduces Fortran's nonlocal
  forces but is not the derivative of the sampled projector (the two differ by
  up to 1e-2 Ry/bohr on Si cells at h = 0.6 bohr; the forces of an isolated
  system use the table);
- the self-consistency correction of `forcloc.f90` is added (Fortran's periodic
  routine has none; it vanishes at convergence); LDA+U (J = 0) and TS-vdW forces
  at Gamma as for clusters;
- with fractional occupations the forces are the derivative of the Mermin free
  energy `E - TS`, not of the printed total energy (as in Fortran).

Symmetrization (`symm_forces.f90`): where the problem carries Fortran's space
group (`Monkhorst_Pack_Shift`, `Symmetrize_Charge_Density`, or a weighted
`Kpoint_List` with forces) the forces are averaged over its operations,
non-symmorphic ones included (`T(r) = U (r - A t)`: the image of an atom is
found modulo lattice vectors as `symm_forces` finds it, the force turns with
`U`), and a relaxation averages its positions over the same operations with
their translations. Otherwise the point operations about the origin that map
the lattice, the grid, the atoms (modulo lattice vectors) and the k-points onto
themselves are used, as before. Molecular dynamics and manual movements use
none, as in PARSEC (`parsec.F90` sets no symmetry for them: with
`Monkhorst_Pack_Shift` their wedge is that of time reversal alone); no torque
is reported.

An irreducible k-point wedge gives the nonlocal forces of its own points
only; summed over the zone they are `(1/N) sum_g U_g F(g^-1 a)` of the wedge
forces, which `Forces.periodic.reduced_kpoint_operations` applies to the
k-dependent terms whatever the `symmetrize` flag (Fortran gets the same sum from
`symm_forces`). A relaxation whose `Move_Flag` the group does not respect is
refused with a reduced k-point set, and molecular dynamics with a weighted
`Kpoint_List`: the structure would leave the group the wedge stands for.

Collinear spin with k-points runs `run_scf_kpoints_spin_polarized` keeping its
orbitals and potentials; the forces sum the nonlocal term over both channels
(one electron per orbital) and the local terms take the total density and the
channel averages, as `forpbc.f90`. With `Double_Grid_Order > 1` the projector
gradients are those of the fine points with their images and the Bloch phase
of the coarse point (`nonloc.F90`).

Relaxation and dynamics use `Relaxation.periodic_backend`: `run_scf` at Gamma
(`run_spin_scf` with collinear spin, started from the periodic spin densities),
`run_scf_kpoints` keeping its orbitals with k-points (with collinear spin
`run_scf_kpoints_spin_polarized`, started from the two channel potentials of
the structure before), and
`SCF.pbc.update_periodic_geometry` after every move (local potential,
projectors, LDA+U projectors, densities, Ewald energy). Atoms that leave the
cell are translated back (`back_to_cell`); the optimizer and the integrator keep
the unwrapped coordinates. The next SCF loop starts from the last mixed
potential with the ionic part replaced (and at Gamma from the saved orbitals).

On a GPU the accelerated command line (`main.py`) runs the forces,
relaxation and dynamics of a bulk cell on Fortran's orthorhombic grid at
Gamma without spin itself, and hands the other periodic inputs (k-points,
collinear spin, LDA+U, TS, wires and slabs, the general grid, ...) to the
reference solver, whose k-point eigenproblems run on the device workers.
For every bulk cell on the orthorhombic grid whose forces are asked for or
whose atoms move, on either route, the ionic fields and their forces are
built on the device (`acceleration/V_ion/periodic_device.py`,
`acceleration/Forces/periodic_device.py`):

- the readable builders sum the local potential and the atomic densities
  over every lattice image out to the end of the radial tables (118 bohr for
  the POTRE files: about 6000 images of each atom of an 8-atom silicon cube,
  70 s for each of the two fields at h 0.45 bohr on the host of an A100 node,
  before the first SCF loop and after every move); the device sums stop where
  the terms vanish: the
  short-ranged piece of the Ewald split `V_a + 2 Z erf(eta r)/r` (with the
  `eta` of the FFT potential of `periodic_setup.py`) where the table of
  `r V_a` is the Coulomb tail and `erfc(eta r)` is below the Ewald tolerance,
  an atomic density where it is below `1e-16` of its peak. The fields equal
  the readable ones to round-off (potential 1e-13 Ry, densities 5e-16);
- the local, core-correction and self-consistency forces are the exact
  derivatives of those fields (short range in one CUDA kernel, the
  reciprocal part from one FFT of the density), so the forces remain the
  derivative of the energy computed;
- at Gamma the nonlocal term is contracted with the orbitals where the
  eigensolver keeps them, with the projector gradients of the lattice images
  (`acceleration/Forces/nonlocal_device.py`); the moved atoms replace the
  projectors of the device Hamiltonian in place, and the next SCF loop
  continues from the saved subspace, as for a cluster.

The G-space local potential keeps the readable fields and forces.
`Cubic_Spline` species keep the fields of the host preparation
(`periodic_setup.py`); their local forces are the derivatives of those fields
(`acceleration/Forces/periodic_spline.py`, since 2026-11-07: the short-ranged
terms on the host over the balls of the potential, the reciprocal part from
one FFT, the self-consistency and core terms in the CUDA kernel above). The
readable routine they replace summed every grid point of every image within
`erfcinv(1e-26)/eta`: 34 s against 3.5 s for the local forces of the 8-atom cube
at h 0.25 on an A100 node, 892 s against 5.2 s for the 64-atom cube at h 0.30.
On Fortran's general (oblique) grid, whose `Cubic_Spline` potential takes the
same split by default (`PARSEC_PERIODIC_OBLIQUE_SPLIT`), the same routine
differentiates it over the lattice balls (the density terms in a lattice
kernel on the device; within 2.2e-8 Ry/bohr of fourth-order differences of
the energy of the potential for a distorted fcc Si16 cell at h 0.30, 4.5e-7 of
the readable forces for a distorted Si2 cell). A plain single point (no forces, no moving atoms) keeps
the readable fields bit for bit on both command lines. The projector
gradients of the forces find their grid points with the grid-ball query of
`periodic_setup.py` (the same matrices). Si8 cube (h 0.64 bohr, laptop RTX
5070): forces equal to the reference solver within 1.3e-12 Ry/bohr, a BFGS
run of four structures within 1.2e-10 bohr (7 s against 272 s). One later
ionic step on an A100 (h 0.45 bohr, Gamma): 149 s before, 0.65 s now for the
8-atom cube; 1473 s and 5.3 s for the 64-atom cube; with 8 k-points the
8-atom cube goes from 151 s to 9.2 s, nearly all of it the k-point SCF
loop.

Validation against Fortran PARSEC (Si2 fcc and Si8 cube at Gamma, Si2 with 8
k-points, spin-polarized Na3, BFGS relaxation and Verlet dynamics of Si2): with
Fortran's G-space local potential emulated and Fortran's projector slopes,
the total forces agree to 2.5e-6 Ry/bohr (Si cells; 6.5e-6 for spin-polarized
Na3) and the relaxation and dynamics trajectories to 1e-6 bohr; the report
The report of the parity round has the numbers. The same comparison
for the space group and the wedge (fcc Si2 with eight operations, half of them
with the translation (1/4, 1/4, 1/4), and an hcp cell, both on reduced
Monkhorst-Pack grids), collinear spin with k-points and the double grid gives
1.3e-7 to 1.8e-6 Ry/bohr, and symmetric BFGS relaxations take Fortran's
structures; numbers in the report of the parity round.

## Spin-orbit coupling and non-collinear magnetism

`Forces/spin_orbit.py` gives the forces of the spinor drivers
(`run_self_consistent_soc`, its collinear-spin form, `run_self_consistent_noncollinear`
and the k-point forms of all three): the ion-ion and local terms with the
total density and the spin-independent part of the potentials, the
Kleinman-Bylander term on both spinor components (one electron per spinor),
and the spin-orbit term, the derivative of the `lzsz`/`lsxy` energy
`sum_n f_n <psi_n|H_SO|psi_n>` with the gradients of the `v_ion`/`v_so`
projectors (Fortran's `forcso` of `soforc.F90`). Every term is the derivative
of the energy the drivers compute (finite differences: 1e-9 Ry/bohr). The
Zeeman terms of collinear spin and of the non-collinear field depend on the
atoms through the density only.

Fortran's spin-orbit term is not the derivative of its energy:
`pseudo.f90` tabulates `dvr_ion` as a value (`(l+1) vlj(l) + l vlj(l+2)`) and
`dvr_so` without the factor `sqrt(h^3)/(2l+1)` of `vsor`, `forcso` multiplies
both overlaps of each product by the KB sign (which then cancels) and
`flsxy` pairs the same `m_l` in both spin components (the Hamiltonian's
`lsxy` shifts it). On AuH (`examples/0d_AuH`) the local and Kleinman-Bylander
tables equal Fortran's, the spin-orbit term on Au is +0.0053 Ry/bohr here and
-0.0580 in Fortran, which a line-by-line port of `soforc.F90` with those
tables reproduces (-0.0577). `forcnloc.f90` sums the Kleinman-Bylander term
over the up components only (times four): equal to both components for
Kramers pairs, not with a magnetization.

Relaxation and dynamics use `Relaxation.spinor_backend`. As in `parsec.F90`
every structure after the first starts from the last mixed potential of the
structure before with its ionic part replaced (`vnew - vion_old + vion_new`
for every spin component; a non-collinear loop keeps its exchange-correlation
field, the direction of the magnetization), and the isolated spin-orbit loops
(with and without collinear spin) also filter the spinors of the structure
before, as PARSEC filters its saved wavefunctions; the k-point and
non-collinear loops keep no subspace between loops and start their first
eigenproblem from random vectors. The first structure of a `Restart_Run` from
a spin-orbit `parsec.dat` starts from the potentials of the file as the single
point does. On AuH (manual movements of 0.03-0.06 bohr) the second and third
structures take 12 and 15 iterations (16 from the atomic densities; PARSEC 11
and 13); after a large BFGS step (0.7 bohr) the warm start costs more
iterations than the atomic densities, in PARSEC as here. The TS-vdW term of a
converged spinor loop is added as for a scalar calculation. The forces are
symmetrized as for a scalar calculation (PARSEC
does so with spin-orbit coupling and non-collinear magnetism too; a magnetic
structure whose moments break the atoms' group needs `Ignore_Symmetry`). The
perturbative correction (`SO_PSP` without `SO_from_scratch`) has spinor
eigenvalues but no spin-orbit total energy here, and PARSEC flags its forces
as inaccurate: forces, relaxation and dynamics are refused with it. With
`Double_Grid_Order > 1` the spin-orbit term uses the gradients of the
double-grid spin-orbit projectors (the fine-point sum of `nonloc.F90`, each
fine sample with the slope of its radial interpolant). With `Polarizability`
and `Skip_Force: false` the forces are those of the last of the seven field
loops (field along -z), as `parsec.F90` computes them after `polar.f90`.
Refused: forces with LDA+U and k-points (scalar or spinor; PARSEC's LDA+U
forces are zero).

## Wires and slabs

`Forces/wire_slab.py` follows `forcloc_wire.f90`/`forcloc_slab.f90` (the
short-range local term, the force on the Gaussian charges in the Hartree field,
the core and self-consistency terms), `forceion_wire.f90`/`ewald_slab.f90`
(ion-ion, with the charged sheets of a slab) and `forcnloc.f90` (projectors
with their images along the periodic directions, Bloch phases for k-points).
With collinear spin the local terms read the total density and the channel
averages, the nonlocal term sums both channels; with spin and k-points it uses
the orbitals the spin k-point loop keeps (`keep_wavefunctions`). LDA+U forces
(J = 0) are added as for clusters (Fortran's are zero).

Symmetrization (`symm_forces.f90`): over the space group the input reader
finds for the wire or slab (`symmetries.f90` with `alatt` = `latt_vec` and
`4 rmax` along the confined directions), non-symmorphic operations included;
operations whose translation moves the cylinder or slab off itself are dropped
(PARSEC applies them). A slab has no group unless bandstruc, spin-orbit or an
NSCF run turns PARSEC's symmetry on; molecular dynamics, manual movements and
Polarizability have none. The averaged forces also complete a k-point set
reduced to the irreducible wedge; as for a bulk cell, the movements such a
wedge cannot follow are refused: molecular dynamics and manual movements with
an unequally weighted `Kpoint_List` when the structure has a group (PARSEC
switches the group off for them; a slab without bands, spin-orbit coupling or
NSCF has none, its list is a plain quadrature), and a relaxation with a
reduced set whose `Move_Flag` a group operation does not respect.

Relaxation and dynamics use `Relaxation.wire_slab_backend` (the periodic
backend with `SCF.wire_slab.update_wire_slab_geometry`): after every move
`back_to_cell` (periodic fractional coordinates into `[-1/2, 1/2]`, the
confined ones kept), `isolat` (an atom outside the cylinder or slab stops the
run, error 291), then the local potential, projectors, LDA+U projectors,
densities and ion-ion terms through the same function as a fresh preparation
(bit for bit). With collinear spin and k-points the spin k-point loop starts
every structure from the two channel potentials of the structure before
(`vnew - vion_old + vion_new`, as the bulk cell).

Validation against Fortran PARSEC (the report of the parity round):
BFGS of a Na4 wire (with and without its D2h group), of an H2 slab (Gamma and a
2x2 shifted grid) and of a three-H slab with collinear spin, of polyacetylene
with 4 shifted k-points; Verlet dynamics of the Na4 wire; manual movements of
`tests/chain_Na`: positions agree to 1e-5 bohr, energies and forces to the
printed digits (LDA; PBE cases differ by the known EASYPBE potential).

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
- **Spin-orbit coupling and non-collinear magnetism**: the accelerated entry point runs these inputs, when they ask for
  forces, with the reference solver and says so. Its pooled k-point solver (periodic and non-collinear inputs) hands the
  orbitals of a k-point to the force pass on the device that solved it: the overlaps with the projectors, the spin-orbit
  projectors and their gradients are sparse products there and only the `(atoms, 3)` result comes back
  (`Forces/spin_orbit.py` is written against the array module of the orbitals; it is tested on CPU arrays through a
  module that offers only CuPy's operations, not yet on a GPU).
- **Periodic cells (Gamma point, no spin)**: forces, relaxation and molecular dynamics run on the accelerated path.
  Moving the atoms replaces the projectors on the device operator and rebuilds the ionic potential, the densities and the
  Ewald terms from the folded positions (`update_periodic_geometry`); the Hartree solve of a periodic cell is the readable
  one. The nonlocal force is contracted on the device that holds the orbitals with the gradients of the periodic
  projectors, as for an isolated structure. The local ionic and core-correction forces
  (`acceleration/Forces/periodic_local.py`) replace the scan of the whole grid for every image and the grid-by-G phase matrix
  of the readable routine with a ball around each image of each atom and one FFT of the density, read at the reciprocal
  vectors of the Ewald series, with the Gaussian width chosen so that the series ends below the Nyquist frequency of the grid
  (the way the fast ionic potential is made). The sums are written against an array module: NumPy on the host, CuPy
  (with `cupyx.scipy.special.erf`) when the calculation ran on a GPU, which then forms the radial slopes, products and sums
  and the FFT; the balls themselves are generated on the host from index ranges. The forces are those of the readable routine
  to 1e-7 Ry/bohr (the splitting of the Ewald sum; 1e-13 for a table cut at 14 bohr). Not accelerated: k-points, spin,
  several nodes (an MPI context is refused for a periodic cell), and the Ewald ion-ion force, which stays on the host.
- **Periodic cells with k-points**: the accelerated command line hands such an input to the reference driver, whose k-point
  solver puts one worker thread on each CUDA device (the output names the device under "Execution resources"). Its
  eigenproblems, the density of each k-point and now the forces run there: the nonlocal force of a k-point is contracted on
  the device that holds its orbitals (`acceleration/Forces/kpoint_nonlocal_device.py`: the overlaps with the complex Bloch
  projectors and their gradients are sparse products on the device, a block of orbitals at a time, and only the `(atoms, 3)`
  array returns; a k-point whose subspace was spilled to the host contracts there), and the local ionic and core-correction
  forces are the ball sums and one FFT of the density of the Gamma-point path, with CuPy when the k-point plan uses a GPU
  (the reference `periodic_local_forces` hands its sums to them once the acceleration package is imported). Not done: the
  row-distributed MPI k-point solver, which has no nonlocal force; and the Ritz subspaces of the k-points, which restart
  from scratch after an ionic move (the Gamma-point loop carries its subspace over), so a relaxation step costs a first solve.

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
| Forces of spin-orbit and non-collinear runs | `forcso`, with a warning that they "may be inaccurate" | The derivative of the energy of the SCF loop, pieces of `forcnloc.f90` and `soforc.F90` rewritten | The spin-flip term of `flsxy` pairs different harmonics than the Hamiltonian; see "Spin-orbit coupling and non-collinear magnetism". |
| Moved atoms with spin-orbit or non-collinear spinors | SCF restarted from the previous wavefunctions | The same: potential (with the field and moments of a non-collinear run) and eigensubspaces | The atomic density extrapolation is for plain spin-orbit coupling only. |
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
printed digit (the previous Fortran tree; since the formats of PARSEC 1.5 the
integer fields are wider and the velocities are `v(t)`).  Against PARSEC 1.5
itself (H2, Langevin with `test_rng` and `Mol_RNG_Seed`, 3 steps and a restart
of 2, both codes with their own SCF): `md_nrg.dat` and `movie.xyz` are the same
text (but for the last digit of one temperature), `md_mech.dat` agrees to its
eighth digit (the forces of the two codes differ at 1e-7), and a restart
reproduces 1.5 only with its fast-forward of the generator
(`tests/unit/test_moldyn.py::Parsec15ExecutableTests` replays 1.5's printed
forces).

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
