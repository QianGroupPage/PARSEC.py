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

Under MPI (`PARSEC_MPI=1`, `main.py --mpi`) the drivers stay replicated on every rank and unaware of MPI;
`make_kpoint_solver` returns one of two solvers. The scalar and collinear-spin k-point drivers get
`acceleration/SCF/kpoint_pools.py`: the ranks form pools (`--kpoint-pools`, else by balance and a memory
estimate), the tasks are dealt to the pools, and a pool shares the basis of each task by columns
(`acceleration/Eigensolvers/chebff_columns.py`: every rank holds all grid rows of its columns and a replica of the
operator, filters without communication, and moves the basis to blocks of rows and back with `Alltoallv` for the
Cholesky QR and the Rayleigh-Ritz step; the dense steps on the pool root); on an orthorhombic grid the local
Bloch term and the recurrence step of the filter run in one kernel (`acceleration/Hamiltonian/bloch_stencil.py`,
read from and checked against the stencil matrices). The jobs with per-task extras (spin-orbit, non-collinear,
forces) get the basis split by blocks of grid rows: `acceleration/Parallel/mpi.py` (the communicator and its
collectives), `acceleration/Hamiltonian/distributed_bloch.py` (row-sliced operators with halo exchange and
projector sums) and `acceleration/SCF/kpoint_mpi.py`; the eigensolver in `chebff_spinor.py` is the same code with
its reductions routed through the operator. Every rank solves the periodic Hartree potential and evaluates the
unpolarized semilocal functionals on its own device (`acceleration/V_xc/periodic_xc.py`; fixed-order sums: the same
fields on all ranks); only rank 0 writes files.

Preparing a periodic cell follows the same rule. `V_ion` keeps the direct definitions: the Ewald local
potential and the projector and atomic-density builders measure the distance of every grid point to every atom
image. `acceleration/V_ion/periodic_setup.py` supplies a grid-ball query (integer index boxes, identical
rows and distances) through `V_ion.ionic_potential.set_support_finder`, the local potential as one FFT
through `V_ion.ewald.set_local_potential_provider`, the superposed atomic densities through
`V_ion.ionic_potential.set_atomic_density_provider` (on the GPU where one is usable, the host sum bit for bit);
importing `parsec_python.acceleration` installs all three.  The Ewald ion-ion energy needs no counterpart since
2026-11-08: `V_ion.ewald.ewald_ion_ion_terms` sums the atom pairs within its cutoff and the structure factors of a
box of reciprocal vectors (0.5 s for 4,096 atoms).
The radial sums reach the end of the radial tables (about 118 bohr), so they visit thousands of images of every
atom in a small cell: the module sums them in blocks of images (broadcast distances per axis, worker threads,
`PARSEC_PERIODIC_SETUP_THREADS`) in the order of the definitions.  By default (since 2026-10-30,
`PARSEC_PERIODIC_SETUP_REACH=physical`) each sum stops where its terms are zero or below round-off, which changes
the last bits (1e-11 Ry in energies); `PARSEC_PERIODIC_SETUP_REACH=table` reaches the end of the tables, bit for
bit the definitions (the default before).
`acceleration/tests/test_periodic_setup.py` checks them against the definitions. The replacements serve the
orthorhombic grid of `Grid.pbc.build_periodic_grid` only; `GeneralPeriodicGrid` and `ConfinedPeriodicGrid`
(subclasses with other point positions) keep the definitions, except the local potential of a `GeneralPeriodicGrid`
cell with a `Cubic_Spline` species, which takes the split of the orthorhombic grid over lattice balls
(`PARSEC_PERIODIC_OBLIQUE_SPLIT`, since 2026-11-07: the direct sum ran into the spline error near the end of the
tables as the cell grew).  For the forces, relaxation and dynamics of
such a cell the accelerated command line installs, for that calculation only, device builders of the local
potential and of the atomic densities (`acceleration/V_ion/periodic_device.py`, through the same provider hook
and `V_ion.ionic_potential.set_atomic_density_provider`) and their derivatives as the local terms of
`Forces.periodic.periodic_local_forces` (`Forces.periodic.set_local_force_provider`,
`acceleration/Forces/periodic_device.py`); the host builders of `periodic_setup.py` are restored afterwards.
The local forces of `Cubic_Spline` species on the orthorhombic grid of the Fortran-derived routines, whose
potential stays the host one, are the derivatives of that potential (`acceleration/Forces/periodic_spline.py`,
installed through the same `Forces.periodic.set_local_force_provider` by `periodic_setup.install`; the device
provider hands those species to it).  The short-ranged sums of the local potential stop where the table of `r V` is
`-2 Z`: to round-off (POTRE), or to the digits of the file (FHIPP, MARTINS_OLD, MARTINS_WANG:
`V_ion.ionic_potential.coulomb_tail_radius`).

The tetrahedron sums of `Create_DOS` follow the same rule: `Output/dos.py` keeps the definition
(`spectrum_sums`, written with the operations NumPy and CuPy share) and a provider hook
(`set_spectrum_provider`); `acceleration/Output/device_dos.py` installs a provider that runs it on the
GPU and sums the pairs in `bincount`'s order, so the `DOS` file is the same bit for bit
(`acceleration/tests/test_device_dos.py`).

The fields of a periodic SCF step follow it too. `Hartree.confined.solve_confined_hartree` (wire and slab:
boundary values of `hartset_wire`/`hartset_slab` and the CG) and `Hartree.pbc.solve_periodic_hartree`
(`Calc_Hartree_G`, the FD-CG of a cell) take a faster solver of the same equations through
`set_confined_hartree_solver` / `set_periodic_hartree_solver`, and the spin functionals of a periodic cell
(`SCF.pbc`, `set_periodic_spin_xc_provider`) a faster evaluator; `acceleration/Hartree/periodic_device.py` and
`acceleration/V_xc/periodic_xc.py` supply the CUDA ones where the k-point plan puts the eigenproblems on a device
(`PARSEC_PERIODIC_HARTREE`, `PARSEC_CUPY_XC`), the plain FD-CG of an orthorhombic 3-D cell among them by default
since 2026-10-30 (`PARSEC_PERIODIC_HARTREE=host` keeps the host CG); the CA and PBE spin functionals of a periodic
cell stay on the host by default. `acceleration/tests/test_periodic_device_fields.py` checks them against the
definitions.

Where a performance path cannot carry a feature of the readable drivers it hands the
combination back by name (`acceleration/tests/test_parity_device_routes.py`): ARPACK and
DIAGLA run on the host route of a device k-point worker or band path
(`SCF.kpoint_solver.device_solver_supports`), the MPI solver refuses them, and the
accelerated CLI refuses under `--mpi` the features that read whole orbitals
(`acceleration/cli.py::_mpi_unsupported_features`).

The flow is a dependency direction, not permission for lower-level modules to
import the SCF driver. Physics components must remain independently callable.

## Responsibilities

| Package | Owns | Does not own |
|---|---|---|
| `Input` | ESDF parsing and validated calculation configuration | grid construction or file output |
| `Grid` | isolated-domain points, masks, shifts, and index maps | kinetic coefficients |
| `Laplacian` | finite-difference coefficients and kinetic application | effective potentials |
| `Pseudopotential` | POTRE, old-format POTR, Martins-Wang POTRW and FHIPP parsing, radial data, Fourier filters, interpolation-ready models | placement on a molecular grid |
| `V_ion` | local potential, KB projectors, atomic density, ion-ion energy | SCF state |
| `Hartree` | isolated boundary values and Poisson solution; the Kronecker-product Poisson solver of batches of pair densities for exact exchange (`kronecker`: 1/r, erfc and erf kernels, multipole boundary on the padded bounding box; reflection-parity classes on the octant of a symmetric box); the Coulomb kernels `K(q + G)` of the exact exchange of periodic cells (`exchange_kernel`: bulk with k-points, slabs padded or truncated, the Gygi-Baldereschi element) | density mixing |
| `V_xc` | LDA (CA/PZ, PW92, X-alpha, Wigner, Hedin-Lundqvist), GGA (PBE, PW91) and asymptotically corrected (LB94, Casida-Salahub) potentials and energy densities, unpolarized and spin-polarized; the Fock exchange operator of orbital blocks, its adaptively compressed form (ACE) and the exchange energy (`exchange_operator`); the hybrid functionals: label registry, exact-exchange fractions and kernels, semilocal parts (`hybrid`, `semilocal`, `semilocal_components`); the exchange of periodic cells (`bloch_exchange`: pair potentials by FFT on the grid box, Bloch factors of a time-reversal-reduced mesh, ACE per k-point) | total-energy assembly |
| `Hamiltonian` | composition and block matrix-vector products | choosing an eigensolver |
| `Eigensolvers` | PARSEC-selected eigensolver control flow and reusable subspaces | occupations or SCF mixing |
| `Occupations` | Fermi level, occupations (searched, or fixed by `occup.in`), and density from orbitals | eigensolution |
| `Mixer` | explicit mixer history and PARSEC SRE metrics; the Anderson history carried through the inner loops of a Fock loop (`carried`, `Exx_Mixer_History: carry`) | convergence policy selection |
| `Energy` | named Rydberg energy components and total energy | text formatting |
| `SCF` | ordering the component calls and applying stopping policy; the Fock loop of the hybrid functionals around the SCF loop (`hybrid`), its controls (`fock_controls`: adaptive inner criterion, carried mixer; `fock_diis`: Anderson extrapolation of the density matrix with a saddle guard); the exchange engine of periodic cells with k-points (`hybrid_periodic`: Bloch orbital sets, ACE per k-point, inner loops on the k-point drivers) | hidden algorithm substitution |
| `Forces` | ion-ion, local, nonlocal forces, their symmetrization and net-force removal | moving atoms |
| `Relaxation` | L-BFGS-B, steepest-descent and manual ionic steps, the ionic loop, relaxation files | electronic structure of a structure |
| `MolecularDynamics` | the integrator of `moldyn.f90`, its input, random numbers and files | energies or forces |
| `Embedding` | PARSEC 1.5's frozen-density embedding (FDET) of a tip in a frozen sample: the sample field on the tip grid, non-additive kinetic functionals, FDET energy terms, the tip sphere, the warm starts of scans, the AFM scan points and `k_ts` | the SCF loop (it calls these), file formats (`Output/pot_dat.py`) |
| `Output` | PARSEC-shaped reporting and optional serialization | numerical decisions |
| `acceleration` | backend selection, symmetry sectors, GPU eigensolvers, native kernels, and timing provenance; the exact exchange of the hybrid functionals on the devices (`Hamiltonian/exchange_device`: ACE per symmetry sector, pair solves on every device, sector pair densities in parity classes without the full grid; `SCF/hybrid`: the device pieces of `SCF.hybrid.run_hybrid_scf`; `SCF/fock_device`: the sector blocks of the outer DIIS and the free-memory test of its window); spin-polarized loops in the symmetry sectors (`SCF/spin_sectors`: the orbit fields of two channels for `SCF.spin_polarized.run_spin_scf`, the occupation guard; `Eigensolvers/symmetry`: one sector subspace per channel) | independent physical formulas or silent accuracy changes |

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
