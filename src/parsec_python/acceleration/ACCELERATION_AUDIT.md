# PARSEC alignment and acceleration audit

This audit covers the implemented scalar, spin-unpolarized, nonperiodic
single-point path with CA/PZ LDA, MARTINS_NEW POTRE pseudopotentials, and
``Double_grid_order=1``. The default native Hartree path applies a validated
totally symmetric wedge from the largest exact commuting signed-permutation
subgroup. On CuPy, the diagonal-reflection action also decomposes orbitals
into all real one-dimensional representations. Exact orbit-stabilizer
character selection covers both free and fixed-point actions; unsupported or
trivial actions retain the full orbital domain in automatic mode.
“Aligned” below means the Python control flow and formulas were checked
against the named PARSEC routines inside `<PARSEC_FORTRAN_ROOT>/src`; it does
not claim support for PARSEC features outside that scope.

The acceleration rule is simple: preserve float64 operators and solver
decisions, keep one-time or tiny work readable, use C++/OpenMP for long host
loops over the real-space grid, and use CuPy only when large arrays can remain
on the GPU across repeated operations.  A GPU port that adds a host/device
round trip is not considered an acceleration.

## File-by-file decisions

| Physical stage | Readable implementation | PARSEC source checked | Alignment within supported scope | Fast implementation and reason |
|---|---|---|---|---|
| Input translation | `parsec_python/Input/parsec_input.py` | `usrinputfile.F90`, `initial.f90` | Restricted ESDF subset; unsupported keywords are rejected rather than guessed | CPU Python. Text parsing is one-time and immaterial. |
| POTRE parsing | `Pseudopotential/potre.py` | `pseudo.f90` | MARTINS_NEW radial data, channels, occupations, cutoff data, valence/core densities | CPU Python/NumPy. Files are small and read once. |
| PARSEC cubic spline | `Pseudopotential/radial_spline.py` | `spline.f90`, callers in `ionpot.f90`, `corecd.f90`, `nonloc.F90` | Origin padding, clamped second derivatives, and `splint` evaluation are retained | Spline setup stays NumPy; repeated atom-grid evaluations use the native radial kernel. |
| Cluster grid and symmetry orbits | `Grid/cluster.py`, `parsec_python.acceleration/Symmetry/axis_reflection.py`, `geometry_cache.py` | `symmetries.f90`, `grid_partition.f90`, `init_var.F90`, `setup.F90` | Same full enclosing grid. The accelerated layer tests all 48 Cartesian signed permutations and retains the largest exact commuting involution subgroup mapping labeled atoms and every active lattice point; normalized orbits satisfy `U.T U=I`. This is broader than axis signs but not yet PARSEC's general point-group implementation. | Full-grid construction remains vectorized NumPy. Exact geometry/grid/tolerance keys persist validated orbit metadata; changed or corrupt entries rebuild. The naphthalene map is exactly 523,984 -> 65,498 points. |
| Orbital representations and state allocation | `Symmetry/representations.py`, `Symmetry/operator_cache.py`, `Eigensolvers/symmetry.py` | `initeigval`/`eigen_sort` in `eigval.F90`, `grid_partition.f90` | All real characters form orthonormal `U_Gamma`; orbit `O_w` is present exactly when its stabilizer lies in the character kernel, with `U_Gamma[i,w]=chi_Gamma(g_i)/sqrt(|O_w|)`. Thus fixed-point sectors may have different dimensions and still satisfy `H_Gamma=U_Gamma.T H U_Gamma`. Initial sector count is `floor(Nstate/Nrep)+nadd`, with a ceil guard, and sectors grow until their final values bracket the global cutoff. All Ritz values are globally stably sorted before occupations. D2h labels use PARSEC's Ag/B1g/B2g/B3g/Au/B1u/B2u/B3u order. | One canonical sparse gather constructs every sector; a SHA-256 exact-content cache reloads GPU-ready stencil metadata and KB factors. Identical topology/maps share allocations; stabilizer-dependent maps retain distinct buffers. The unused full-grid CUDA Hamiltonian is deferred. One GPU stays serialized by default; multiple visible GPUs receive sectors round-robin and run concurrently, with per-device operators and saved state. |
| Finite-difference coefficients and domain boundary | `Laplacian/finite_difference.py` | `fornberg.f90`, `init_var.F90`, `matvecB.f90z` | Centered even-order stencil, compressed-grid row order, and omitted exterior orbital neighbors (zero Dirichlet boundary) | Native C++ builds canonical CSR. CUDA transposes the short rows to coalesced `neighbor[slot,row]` int32 storage with uint8 codes into an exact float64 palette, while retaining per-row summation order. |
| Local ionic potential | `V_ion/ionic_potential.py` | `ionpot.f90` | Per-atom distance, default interpolation of `r*V(r)`, optional direct spline, switch at `rs(ns-1)`, and `-2Z/r` Rydberg tail | Cached-coordinate C++/OpenMP radial evaluator. Full naphthalene maximum error versus NumPy: `3.55e-15 Ry`. |
| Initial valence density | `V_ion/ionic_potential.py` | `initchrg.f90` | Spherical atomic superposition, linear stored-density interpolation, radial-wavefunction reconstruction, cutoff, and final electron normalization | Same cached C++/OpenMP atom-grid evaluator; normalization remains one NumPy reduction. |
| NLCC core density | `V_ion/ionic_potential.py` | `corecd.f90` | Same stored radial field, cutoff, and optional PARSEC spline for supported nonnegative POTRE data | Same native radial evaluator. Species without NLCC are skipped. |
| KB nonlocal construction | `V_ion/ionic_potential.py` | `nonloc.F90` | `Delta V_l u_l/r`, radial denominator sign, channel support, real harmonics, Euclidean-grid normalization, and PARSEC column order for `Double_grid_order=1` | C++/OpenMP samples support/radial/harmonic values; Python visibly assembles sparse CSC columns. Full naphthalene support, labels, and signs were identical; maximum value error `3.47e-18`. |
| KB nonlocal application | `V_ion/NonlocalProjectorOperator.apply`, `Symmetry/representations.py`, `backends/cupy_projectors.py`, `cupy_stencil_major.py`, and `Hamiltonian/operator.py` | `nonloc.F90`, `matvecB.f90z` | Matrix-free `B diag(sign(D)) B.T`; sector factors are exactly `B_Gamma=U_Gamma.T B`, so no dense grid-by-grid nonlocal matrix is formed | Production sectors upload raw canonical CSR factors without constructing cuSPARSE objects. Short `diag(signs) B.T X` rows retain serial source order; rows of at least 256 entries use a deterministic CUDA tree reduction. The large `B C` scatter is fused into the stencil/local/recurrence kernel. cuSPARSE, C++/SciPy, and environment-controlled fallbacks remain available. |
| Hartree boundary and Poisson solve | `Hartree/poisson.py`, `parsec_python.acceleration/Hartree/native_boundary.py`, `symmetry_poisson.py` | `rho_hart.F90`, `hartset.F90`, `hpotcg.F90`, `grid_partition.f90` | Same multipole Dirichlet boundary, boundary-to-interior correction, `8*pi*rho`, warm start, CG tolerance, matvec budget, and final true residual. For a proven symmetry group, `A_w=U.T A U`, `b_w=U.T b`, and the result is expanded by `U`, matching PARSEC's totally symmetric Hartree representation. | Native 0.5 post-release persistently caches orbit-summed harmonic/boundary geometry and constructs normalized `b_w` directly from one density value per orbit. Compact deterministic CG stays on the wedge and fuses `A p` with `p dot A p` while preserving canonical row and block-reduction order. A two-step chronological RHS/solution predictor improves the initial guess after two solves without changing the converged equation. Repeated boundary/CG vector loops size their OpenMP team from the grid workload. Full-grid native/Python and GPU-CG paths remain fallbacks. |
| CA/PZ LDA | `V_xc/ca_lda.py`, `parsec_python.acceleration/V_xc/native_ca_lda.py` | `exc_nspn.f90` | Same exchange and the `rs>=1`/`rs<1` Ceperley–Alder/Perdew–Zunger branches, Rydberg factors, zero-density behavior, and NLCC input | Cached-core C++/OpenMP evaluator. Under symmetry it consumes compact physical orbit values and uses exact integer multiplicities for `E_xc`; array errors remain at float64 roundoff. |
| Hamiltonian assembly | `Hamiltonian/operator.py`, `Symmetry/representations.py`, `SCF/single_point.py` | `matvecB.f90z`, `parsec.F90` | `H=-nabla_FD^2+V_ion,local+V_H+V_xc+V_NL`; only the local field changes during SCF. Reflection sectors use the exact projection `H_Gamma=U_Gamma.T H U_Gamma`. | CUDA uses coalesced stencil-major metadata and fuses kinetic, local diagonal, optional nonlocal image, and normalized Chebyshev recurrence into one grid pass. No dense Hamiltonian is formed. |
| Lanczos spectral bounds | `Eigensolvers/spectral_bounds.py` | `eigval.F90`, CHEBFF/CHEBDAV/SUBSPACE sources | PARSEC step limits and scalar tridiagonal bound policy are retained | Large vectors, H applications, and recurrence scalars stay on the CUDA stream. One batched transfer returns the at-most 8-by-8 tridiagonal and breakdown decisions to CPU; the identical first residual norm is reused. Concurrent sector bounds remain an opt-in experiment because they measured slower. |
| CHEBFF | `Eigensolvers/chebff.py`, `chebyshev.py`, `lapack_random.py` | `chebff.f90z`, `dgks.f90z`, LAPACK `DLARNV` | First-SCF fixed filter cycles, degree, bounds, block order, orthogonalization, lower-triangle Ritz solve, and exact 48-bit initial random stream | Device-resident CuPy. CHEBFF skips residual formation exactly as PARSEC does. The host DLARNV stream uses 16,384 exact skip-ahead lanes instead of millions of Python iterations. Cross-block filtering remains opt-in because it was slower on this GPU. |
| CHEBDAV | `Eigensolvers/chebdav.py`, `orthogonalize.py`, `lapack_random.py`, `symmetry.py`, `backends/cupy_orthogonalize.py` | `chebdav.f90z`, `dgks.f90z`, LAPACK `DLARNV` | Non-BETA real path: filtered expansion, contiguous locking, active-space limits, inner/outer restarts, approximate cleanup, bit-exact random sequence, and an independent solve per accepted representation | All wedge/full-grid bases, H images, Ritz work, and residuals use CuPy float64; only control scalars cross to host. Exact vectorized skip-ahead removes scalar Python random generation. At 100,000 rows or more, appended blocks use audited FP64 block-CGS2/device-MGS2 with Householder and literal-MGS fallbacks. A contiguous full-workspace GEMM forms prefix coefficients; a six-vector CUDA row kernel applies only the active prefix and subtraction. The incremental Ritz projection also uses the contiguous complete workspace and consumes only active rows. Small sectors retain source MGS. |
| Later SUBSPACE | `Eigensolvers/subspace.py`, `chebyshev.py`, `rayleigh_ritz.py`, `small_dense.py`, `symmetry.py`, `backends/cupy_mixed_precision.py` | `subspace.f90z`, `eigval.F90` | One later-SCF filter/orthogonalization/Ritz pass per saved sector, degree adaptation, saved buffered state, and global eigenvalue merge | CuPy-resident per-sector state; projected symmetric solves of order at most 64 use faster host LAPACK by default. The production SCF skips diagnostic Ritz residual formation; modular calls retain it. Small complete bases retain audited PARSEC MGS. Large bases use an audited generalized Cholesky-whitened Ritz solve, persistent column-major `H X`, and robust Householder-QR fallback. Sectors with at least 100,000 rows automatically use FP32 only for the subspace-generating filter, returning to FP64 before Ritz/SCF; FP64 DSYRK computes only the overlap triangle. Both have explicit FP64 controls/fallbacks. |
| Occupations | `Occupations/fermi_dirac.py` | `flevel.f90` | Spin-degenerate `[0,1]` occupations, finite-temperature count solve, and zero-temperature degeneracy handling | CPU NumPy. The state vector is small; CUDA launch/transfer overhead would dominate. |
| Orbital density | `Occupations/fermi_dirac.py`, `parsec_python.acceleration/Occupations/symmetry_density.py` | `newrho.F90` | `rho_i=(2/h^3) sum_n f_n |q_in|^2` for Euclidean-normalized orbitals. For real one-dimensional representations, phase squares are one, so wedge density is algebraically identical on every orbit image. | A fused CUDA row kernel consumes resident normalized wedge orbitals and downloads only compact orbit values. Signed full-grid orbitals and scalar result fields are materialized once at final output. |
| Anderson mixing and SRE | `Mixer/anderson.py`, `parsec_python.acceleration/SCF/symmetry_fields.py` | `anderson.f90`, `getsre.f90` | Same potential residual, history differences, small dense solve, linear parameter, restart schedule, weighted SRE, and plain SRE | With orbital symmetry, density, all scalar potentials, history, residuals, and Gram products stay one value per orbit with exact multiplicity weights through SCF. Full-grid NumPy remains the fallback. |
| Total energy | `Energy/total_energy.py`, `parsec_python.acceleration/SCF/symmetry_fields.py` | `totnrg.f90` | Same band energy, old input HXC subtraction, new Hartree half correction, new XC energy, electron-ion diagnostic, and ion-ion term | Density-potential integrals use the same multiplicity-weighted orbit quadrature when sectors are active; the original full-grid dots remain the default otherwise. |
| SCF control | `SCF/single_point.py` | `parsec.F90`, `eigval.F90`, `newrho.F90`, `getsre.f90`, `totnrg.f90` | Initial SAD/Hartree/XC, first solver then saved SUBSPACE, potential mixing, strict residual stop, and near-convergence filter-degree reduction | Python orchestration deliberately remains readable; expensive callables are substituted component by component. |
| Output/archive | `Output/` | formatting sites in `parsec.F90`, `eigval.F90`, `totnrg.f90` | PARSEC-shaped report with explicit PARSEC-PYTHON/backend provenance and one-based representation labels after the global sort | CPU text/NPZ output; not performance relevant. |

## Unsupported PARSEC paths

The accelerated package does not silently approximate general point-group
representations. It supports exact commuting Cartesian signed permutations
for scalar/Hartree fields and all real orbital representations of the
diagonal-reflection subgroup, including fixed-point stabilizers. Unsupported
generalized-operation cases retain the broader totally symmetric Hartree
wedge while orbitals use the exact diagonal subgroup or full grid in `auto`.
It does not
support periodic boundary conditions, spin polarization or spin orbit,
complex wavefunctions, `Double_grid_order>1`, point charges, forces, dipoles,
LDA+U, or alternate XC functionals. MPI reduction order and general Fortran
point-group conventions are not claimed. Inputs selecting an unsupported
algorithm must fail explicitly.

## Why the recorded 28-rank PARSEC calculation is fast

The naphthalene comparison was not a 28-CPU calculation on the same full
Python matrix. PARSEC reports all of the following in the recorded output:

- an Abelian `D2h` subgroup with eight operations;
- 523,984 full points but only 65,498 points in the irreducible wedge;
- 28 MPI ranks split into four independent groups of seven ranks;
- eight representation eigenproblems, with four processed concurrently;
- a neighborhood graph communicator for stencil halos and nonblocking
  nonlocal-projector all-reduction.

Inside `matvecB.f90z`, each rank owns a spatial row partition. It prebuffers
orbital halo rows, starts the nonlocal dot-product reduction, exchanges only
stencil-neighbor halos, computes independent local work while communication
is outstanding, and then finishes the nonlocal image. `chebff.f90z` and
`subspace.f90z` keep this distribution through filtering and reduce only the
small projected matrices. `hpotcg.F90` likewise solves the totally symmetric
Poisson problem on distributed wedge rows.

The build itself is conventional (`-O2`, MPICH, BLAS/LAPACK/OpenBLAS). The
large advantage comes from symmetry, spatial ownership, concurrent
representations, and overlapped collectives—not a special Fortran arithmetic
instruction. The Python GPU path now constructs
`H_Gamma = U_Gamma.T H U_Gamma`, retains independent sector subspaces, and
merges Ritz values before occupations. A nonblocking CUDA-stream scheduler
can overlap independent sectors while preserving fixed-order collection, but
2- and 8-worker naphthalene measurements were slower than serialization
because each filter already saturates this GPU. Single-GPU execution therefore
remains sequential; multiple visible GPUs receive sectors round-robin and run
concurrently, the implemented device-level analogue of PARSEC's independent
MPI representation groups.
Adding `mpi4py` around full-grid arrays would still not reproduce PARSEC's
domain-decomposed communication overlap.

## Remaining acceleration opportunities, in priority order

| Area/file | Measured status after this audit | Decision |
|---|---|---|
| `Eigensolvers/symmetry.py`, `chebdav.py`, `subspace.py` | Multi-GPU round-robin sector ownership is implemented; one GPU remains serialized because stream overlap was slower. The eight adaptive CHEBDAV/SUBSPACE streams do not generally have equal shapes or filter degrees. | Validate scaling on a real multi-GPU host. A future batched implementation must group only equal-shaped/equal-degree work and preserve per-sector restart, locking, random-stream, and stable-merge decisions. |
| `Eigensolvers/orthogonalize.py`, `small_dense.py`, `backends/cupy_orthogonalize.py` | Host LAPACK for projected order <=64 remains faster. Complete-subspace MGS speculates only on the common first PARSEC pass and reruns the untouched literal path on a failed audit. Large CHEBDAV blocks use full-workspace coefficient/Ritz GEMMs plus the fused active-prefix CUDA update. One-prefix CGS, a Fortran-order basis, 12-vector blocks, QR, and CholeskyQR2 were slower or triggered audited fallbacks. | Keep the retained audited paths. The next credible step is equal-shape representation-wide batched projected solves; it must preserve each sector's restart and stable ordering. |
| Whole Chebyshev loop | The stencil/local/KB scatter recurrence is fused, while orthogonalization, Ritz rotations, and adaptive control still launch separately. | CUDA graph capture or a larger custom extension may help only after grouping fixed-shape cycles. Dynamic CHEBDAV restarts/bounds and changing pointers make unconditional capture unsafe; no physics-changing graph is enabled. |
| `backends/cupy_stencil_major.py`, `cupy_projectors.py`, `cupy_mixed_precision.py` | The row action is coalesced and recurrence-fused. A SPARC-style padded shared-memory Cartesian tile was 1.9x slower on the compact sphere; 8/12-orbital FP64 row kernels and 128-thread blocks were also slower and were removed. Production sectors own raw CSR factors; cuSPARSE did not improve wide projector contraction and slowed initial CHEBDAV. | Keep the six-orbital, 256-thread canonical sparse path. The size-gated FP32 later filter is the retained consumer-GPU throughput optimization; its FP64 Ritz boundary and full-SCF regressions are mandatory. |
| `Hartree/native_boundary.py`, native CG | Direct orbit RHS is persistent-cache backed; wedge CG fuses `A p` and `p dot A p`. Chronological initialization reduced the Si28H36 median Hartree subtotal from 1.980 to 1.856 s. Jacobi is exactly a scalar identity for that constant-diagonal operator, and PyAMG increased total time to 51.95 s. | Keep native CG plus the exact converged chronological start. A GPU persistent-CG kernel could help only if it beats OpenMP without transfers or changed stopping rules. |
| XC, density, mixing, energy | All scalar fields now remain compact end-to-end; naphthalene mixing/energy is about 0.063--0.071 s and native XC about 0.005 s over SCF. | These stages are below the profitable GPU-transfer threshold. Retain native/NumPy weighted wedge work unless a future all-device Hartree path removes the host boundary. |
| Python/CPU threading and MPI | CUDA discovery is process-cached and long host loops already execute in OpenMP outside the GIL. | Python 3.14 free-threading does not accelerate CuPy, BLAS, or pybind/OpenMP kernels. MPI becomes useful only with PARSEC-style spatial ownership and halo/collective overlap, not replicated Python arrays. |

## Measured consequences on this workstation

- A paper-guided Si28H36 profile identified later filtering and generalized
  Ritz work as the dominant GPU stages. The retained size-gated FP32 filter
  plus FP64 DSYRK reduced a fresh complete run from 71.80 s to 61.11 s and
  diagonalization from 62.80 s to 51.58 s. All 14 printed SCF energies and
  the final `-263.51147864 Ry` matched the FP64 baseline. FP32 `H X` inside
  Ritz was rejected after shifting the final energy by `5.85e-4 Ry`;
  shared-memory/padded-domain stencils, wider FP64 row kernels, 128-thread
  launches, and cuSPARSE projector projection were all measured slower and
  removed.

- The fixed-point symmetry audit decomposed the 361,096-point Si28H36 grid
  into exact sectors of 182,718 and 178,378 points. Stabilizer-aware sectors,
  contiguous-workspace CHEBDAV prefix projection, and chronological Hartree
  initialization reduced the matched median/default result from 25.17 s to
  22.68 s total (20.156 to 19.164 s SCF). The final energy changed from
  `-263.51147864` to `-263.51147823 Ry` (`4.1e-7 Ry`), and the largest final
  printed eigenvalue change was `1.11e-4 Ry` at the configured `1e-4`
  eigensolver tolerance. Fortran-order Davidson storage, alternate in-place
  GEMM, single-GPU sector streams, and PyAMG were slower or unstable and were
  removed from the production path. The accelerated suite passed 137 tests
  with two optional-runtime skips.

- Naphthalene static physical preparation: 2.23 s reference versus 0.38 s
  with native finite difference and radial ionic setup.  The accelerated
  number excludes one-time CUDA backend/JIT initialization.
- The compact CUDA kinetic+local kernel was 3.5--4.8x faster than generic
  CuPy CSR for representative 6--43-vector blocks, with maximum error below
  `2.9e-13` on the full naphthalene stencil.
- Keeping orbitals resident reduced the complete ten-call density-build cost
  to about 0.014 s; only densities and the final requested orbitals download.
- The compact-Hamiltonian/device-density naphthalene run took 22.93 s and
  ended at `-123.37042729 Ry`, versus `-123.37042748 Ry` in the recorded
  28-rank Fortran run.  The earlier accelerated path took 31.66 s on the same
  input, and the recorded Fortran wall time was 19.63 s.
- The subsequent coalesced stencil-major/fused-recurrence path preserved the
  full printed energy trajectory and final `-123.37042729 Ry` result. Against
  the installed-native 0.3 baseline it reduced diagonalization from 9.66 s to
  7.10 s, SCF from 19.68 s to 16.11 s, and complete process wall time from
  24.41 s to 21.16 s.
- The exact Cartesian-reflection Hartree wedge reduced the complete Hartree
  subtotal from 7.36 s to 0.96 s and SCF from 15.83 s to 8.54 s. Full process
  wall time was 13.68 s, already below the recorded 28-rank PARSEC run.
- Replacing scalar Python DLARNV generation with the bit-exact skip-ahead
  implementation retained every printed energy and reduced the latest run to
  6.52 s diagonalization and 8.12--8.16 s SCF. Complete cold-process wall was
  11.43 s in the best run and 13.13 s in a repeated final-provenance run; CUDA
  initialization accounts for most of that variation. The final energy
  remained `-123.37042729 Ry`; recorded PARSEC is
  `-123.37042748 Ry` with 18.51 s SCF.
- The completed eight-sector D2h CuPy path produced the same one-based
  representation label as PARSEC for every one of the 30 final naphthalene
  states and reached the same final active counts, `9 8 8 9 7 9 9 9`. It
  converged to `-123.37042737 Ry`, with 4.06--4.21 s diagonalization,
  5.68--5.86 s SCF, and 12.42--12.55 s cold complete-process time. The
  recorded Fortran calculation is `-123.37042748 Ry`, 18.51 s SCF, and
  19.63 s total wall time.
- Removing the unused full-grid CUDA allocation, constructing all reduced
  operators in one canonical sparse pass, content-caching the exact static
  transforms, and sharing identical compiled stencil kernels retained
  `-123.37042737 Ry` and all representation labels. The GPU-ready v3 cache
  stores one common neighbor topology and representation-dependent codes,
  palettes, and KB factors. The naphthalene entry is about 29 MB instead of
  211 MB, and a measured load fell from about 0.131 s to 0.035 s. Any exact
  operator, projector, grid-map, sign, or character-phase change invalidates
  it.
- Retaining selected orbitals and scalar fields on the wedge kept the complete
  printed SCF energy trajectory unchanged. The ten
  repeated occupation/density stages fell from about 0.025 s to
  0.012--0.013 s, with an eightfold smaller repeated orbital workspace; full
  signed wavefunctions are reconstructed once for final output.
- Native 0.4 direct-wedge multipoles/RHS, fused CUDA KB scatter, and
  multiplicity-weighted SCF scalar algebra again preserved all ten printed
  naphthalene energies, final `-123.37042737 Ry`, and sector counts
  `9 8 8 9 7 9 9 9`. Against the immediately preceding v3 profile,
  diagonalization fell from 4.326 s to 3.868 s, Hartree from 1.031 s to
  0.611 s, mixing/energy from 0.363 s to 0.173 s, and SCF from 5.932 s to
  4.825 s. A warm geometry/phase/operator-cache process took 6.69 s versus
  10.68 s previously. Geometry and representation construction are now
  exact-key cached; cold runs still rebuild and validate them.
- Concurrent nonblocking streams for only the eight one-vector Lanczos bounds
  preserved every printed value but increased diagonalization to 4.099 s and
  SCF to 5.089 s. `PARSEC_CUPY_COLLECTIVE_LANCZOS=1` retains that experiment;
  the measured-fast default is off.
- Native 0.5 persistent Hartree geometry, fused CG matrix-vector/dot work,
  compact end-to-end scalar fields, cached CUDA discovery, small host LAPACK,
  the custom canonical-order CUDA projector, broader scalar symmetry, and
  multi-GPU sector ownership preserved every one of the ten naphthalene energy
  values and final `-123.37042737 Ry`. In a paired projector comparison,
  diagonalization fell from 3.842 s to 3.622 s and SCF from 4.585 s to
  4.403 s. The final default validation measured 3.615 s diagonalization,
  0.627 s Hartree, 0.063 s mixing/energy, and 4.380 s SCF; complete-process
  wall time was 8.29 s and remains sensitive to CUDA process initialization.
- The final structure/synchronization pass retained all ten printed energies,
  final `-123.37042737 Ry`, and sector counts `9 8 8 9 7 9 9 9`. Raw
  projector factors removed cold cuSPARSE setup; device-resident Lanczos
  scalars removed per-step barriers; audited complete-basis MGS, shared local
  fields, adaptive projector reductions, 2,048-lane DLARNV tiling, and lazy
  diagnostic/SciPy-optimize imports removed smaller repeated costs. The final
  cache-hit run measured 3.502 s diagonalization, 0.601 s Hartree, 4.235 s
  SCF, 8.39 s internal wall, and 9.08 s complete-process wall. The full
  accelerated suite passed 112 tests (2 intentional CUDA-availability skips),
  and the reference suite passed 77 tests.
- The 8.39 s single-run total is 0.10 s slower than the saved v12 8.29 s
  total, so it is not claimed as an end-to-end speedup. The calculation itself
  was faster (4.235 s versus 4.380 s SCF); phase instrumentation assigned the
  reversal to pre-SCF CUDA/process startup. Three warm-cache current-code
  repeats measured 7.06, 7.17, and 7.29 s internally and 7.74, 7.82, and
  7.93 s for the complete process. Performance decisions therefore require
  interleaved repeated medians under the same cache and power conditions.
- CUDA driver discovery was then overlapped with the independent CPU
  grid/ionic/finite-difference preparation. The final automatic-symmetry,
  canonical-projector validation measured 7.99, 6.03, and 6.39 s internally
  (median 6.39 s) and 8.62, 6.65, and 7.01 s for the complete process. All 300
  printed eigenvalue rows, all ten energies, final `-123.37042737 Ry`, and
  sector counts `9 8 8 9 7 9 9 9` exactly matched the validated v10 output.
  The accelerated suite now passes 113 tests (2 intentional availability
  skips); the reference suite passes 77 tests.
- CHEBDAV now reuses Ritz eigenvalues already present on the host after its
  small LAPACK solve instead of issuing repeated one-scalar device transfers.
  In an interleaved three-run A/B on the same naphthalene cache, internal
  median wall time changed from 6.57 s to 6.47 s; every one of the 300 printed
  eigenvalue rows and 80 printed physical energy rows remained exactly equal
  to the validated v10 output.
- Native Hartree boundary geometry is now prepared concurrently with the
  independent GPU orbital setup. In an interleaved warm A/B, complete internal
  median time changed from about 6.37 s inline to 6.26 s overlapped, while the
  main thread waited only about 16 microseconds at the join. All three enabled
  runs exactly retained the 300 eigenvalue rows, 80 energy rows, final
  `-123.37042737 Ry`, and sector counts `9 8 8 9 7 9 9 9`. Both optimizations
  are general execution-schedule/synchronization changes: neither alters a
  real-space operator, eigensolver tolerance, SCF equation, or summation
  order.
- Exact resident reference/operator reuse, exact NNZ caching, workload-sized
  native Hartree/CA-LDA teams, and serial tiny host LAPACK reduced the fixed
  `parsec_resident_optimized.out` naphthalene baseline from 5.00 s to 4.21,
  4.23, and 4.19 s in three warm runs. All ten printed energies were
  identical, the 300 printed eigenvalues changed by at most `2.0e-10 Ry`, and
  the final energy and sector allocation remained `-123.37042737 Ry` and
  `9 8 8 9 7 9 9 9`. The 65,498-point wedge selected eight native workers;
  larger grids automatically scale toward the configured OpenMP maximum.
- The 361,096-point, 80-state Si28H36 full-grid case exposed a different
  regime from the small symmetry sectors above. Its full-grid projector path
  had failed to pass the measured maximum row support (1,187 values) into the
  adaptive CUDA reduction policy; correcting that made the projector dot
  3.43 times faster and the fused Hamiltonian action 1.08 times faster. More
  importantly, one later-SUBSPACE profile assigned 7.48 s to sequential MGS,
  versus 1.13 s for stable Householder QR. The size-adaptive orthogonalizer
  reduced the saved run from 196.24 to 82.04 s total wall time and from
  193.166 to 79.349 s SCF time. All 14 printed total energies were identical,
  the final value remained `-263.51147864 Ry`, and the maximum difference in
  1,120 printed eigenvalue rows was `1.0e-10 Ry`. Large-block Chebyshev
  batching was measured but rejected: 0.489 s versus 0.260 s for the ordinary
  block schedule on this GPU.
- A subsequent large-basis algorithm pass replaced separate tall QR plus
  ordinary Ritz with the equivalent generalized problem
  `(X.T H X) C = (X.T X) C epsilon`. The filtered overlap condition was about
  `5.1e5`, its coefficient orthogonality error was below `4e-15`, and the
  generalized/QR Ritz values differed by at most `1.7e-11 Ry` in the isolated
  stage test. Keeping the filtered vectors column-major and reusing one
  `N x states` `H X` workspace reduced the complete saved Si28H36 run from
  82.04 to 69.46 s, SCF from 79.349 to 66.390 s, and later SUBSPACE from
  45.548 to 32.874 s. All 14 printed energies remained identical, final
  energy remained `-263.51147864 Ry`, and 1,120 printed eigenvalues differed
  by at most `1.0e-10 Ry`. Ill-conditioned overlaps automatically use the
  stable QR route and remember that fallback for later iterations.
- The opposite naphthalene regime remained on its small-sector MGS policy.
  A post-change run reproduced all ten printed energies exactly, retained
  `-123.37042737 Ry`, and differed from the saved validation by at most
  `1.0e-10 Ry` across 300 printed eigenvalue rows. Its measured internal wall
  time was 5.31 s; as with earlier short runs, startup and power-state noise
  make the physics parity more meaningful than one timing sample.

These timings are profiles, not portable promises.  GPU model, memory
bandwidth, CUDA/CuPy versions, OpenMP runtime, and thread affinity must be
recorded with future comparisons.
