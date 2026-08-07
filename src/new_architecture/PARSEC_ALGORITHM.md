# PARSEC Isolated Real-Space Single-Point Algorithm

## Review scope and conventions

This document describes the scalar, spin-unpolarized, nonperiodic single-point
path in WSL PARSEC commit
`f432777750c2efc633adeb26faff641500b39b4a`.

PARSEC's conventions must be kept together:

- length: bohr
- density: electrons/bohr³
- potential, eigenvalue, and total energy: Rydberg
- orbital vectors: Euclidean norm one on the discrete grid
- kinetic operator: `-∇²`, not the Hartree-unit `-1/2 ∇²`
- local ionic tail: `-2 Z/r`
- Hartree equation: `-∇² V_H = 8π rho`

Mixing either Hartree- and Rydberg-unit terms produces factor-of-two errors in
every major energy component.

## Driver order

The relevant top-level sequence is in `src/parsec.F90`:

| Order | Routine | Purpose |
|---:|---|---|
| 1 | `usrinput` | Read geometry, domain, grid, solver, and SCF settings |
| 2 | `pseudo` | Read and condition radial pseudopotentials |
| 3 | `initial` | Electron count and finite-cluster recentering |
| 4 | `symmetries` | Point group and maximal Abelian subgroup |
| 5 | `init_var` | Grid dimensions and finite-difference coefficients |
| 6 | `setup` | Active points, wedge maps, neighbors, MPI partition |
| 7 | `create_eigen_solver` | Copy stencil and solver settings |
| 8 | `nonloc` | Sparse Kleinman-Bylander projectors |
| 9 | `corecd` | Optional nonlinear core density |
| 10 | `ionpot` | Static local ionic potential |
| 11 | `forceion` | Ion-ion energy and force |
| 12 | `initchrg` | Superposed atomic starting density |
| 13 | `rho_har` | Initial Hartree and XC potentials |
| 14 | SCF loop | `eigval -> flevel -> newrho -> rho_har -> totnrg -> mixer` |

`upot.F90` is DFT+U, despite its name, and is inactive for plain CA-LDA.
Periodic local-potential compensation and Fourier filtering are also absent
from the normal confined path.

## Geometry and finite domain

### Recentering

For a new nonperiodic run, `initial.f90:133-198` computes

```text
shift = 0.5 * (coordinate_min + coordinate_max)
```

component by component. If `norm(shift) > 0.1 bohr`, the atoms and point charges
are translated by `-shift`. This is the midpoint of the coordinate bounding
box, not the center of mass. Restart/manual-movement paths bypass it.

### Domain shapes

`usrinputfile.F90:537-718` supports:

- sphere: `x²+y²+z² <= R²`
- ellipsoid: `(x/Rx)²+(y/Ry)²+(z/Rz)² <= 1`
- cylinder: radial distance `<= Rc` and axial distance `<= L/2`
- box: `abs(x)<=Lx/2`, and likewise for y and z

Surface points are active because `outside_domain` uses strict `>` tests
(`grid_partition.f90:795-868`). The enclosing `rmax` is the sphere radius,
largest ellipsoid semiaxis, cylinder half-diagonal, or box half-diagonal.

### Base Cartesian lattice

For a confined calculation, `init_var.F90:96-109` sets isotropic spacing
`step=(h,h,h)` and

```text
n1 = n2 = n3 = floor(2*rmax/h) + 2
```

The unpadded integer bounds in `grid_partition.f90:101-166` are

```text
lo = -(n // 2)
hi = n + lo - 1
r(i,j,k) = h * ((i,j,k) + shift)
```

Fortran integer division truncates toward zero. The distinction matters when
translating a negative odd value to a language whose floor division behaves
differently.

The default shift is `(0.5,0.5,0.5)` grid spacings
(`usrinputfile.F90:781-794`). A source quirk couples
`Ignore_Symmetry=true` to shift `(0,0,0)`. Shift and symmetry are scientifically
separate choices even though that input flag changes both in this commit.

Points are visited with descending x, y, and z integer indices, z fastest.
Only points passing the domain predicate receive active indices.

### Symmetry reduction

For a spherical cluster, PARSEC normally finds the point group, selects a
maximal Abelian subgroup, and stores only an irreducible wedge. Full-grid
neighbors are mapped back to wedge rows with a symmetry transformation and the
appropriate real character (`symmetries.f90`, `grid_partition.f90:268-428`,
`matvecB.f90z:748-822`).

With identity symmetry, the wedge is the complete active grid and every
character is one. This gives the ordinary Cartesian operator and is the clean
reference form for a modular implementation.

## Centered finite differences

`Expansion_Order` is an even integer from 2 through 20; its default is 12
(`usrinputfile.F90:1241-1266`). Internally,

```text
N = Expansion_Order / 2
```

is the number of neighbors on each side along each Cartesian axis. It is not
the `Double_grid_order`, which controls atom-centered Ono-Hirose integration.

For stencil radius `N`, `fornberg.f90` supplies

```text
D2 f_i = [c0 f_i + sum(j=1..N) cj (f_i-j + f_i+j)] / h²

cj = 2 (-1)^(j+1) (N!)² /
     [j² (N-j)! (N+j)!]

c0 = -2 sum(j=1..N) cj
```

The truncation error is `O(h^(2N))`. `init_var.F90:390-420` negates and scales
these coefficients, so the stored operator is `-∇²`.

Default order 12 uses `N=6`: a 13-point one-dimensional stencil or a 37-point
three-dimensional axial star. The dimensionless second-derivative
off-diagonals are

```text
c1..c6 =
 1.714285714285714
-0.267857142857143
 0.052910052910053
-0.008928571428571
 0.001038961038961
-0.000060125060125
```

and `c0=-2.982777777777778`.

### Wavefunction boundary

PARSEC retains the same full centered stencil near a curved domain boundary.
Every inactive or halo neighbor maps to a sentinel whose wavefunction value is
zero (`grid_partition.f90:701-715`, `setup.F90:205-267`,
`matvecB.f90z:261-295`). It does not use a one-sided stencil and does not
interpolate to the geometric surface:

```text
psi(outside active domain) = 0
```

The operator remains real symmetric. Formal high order is lost at the outer
few layers, so the physical domain must be large enough that orbitals have
decayed there.

## Martins-new pseudopotentials

### POTRE layout

`pseudo.f90:360-522` reads the `MARTINS_NEW` `_POTRE.DAT` format:

1. element, XC label, relativity label, core-correction marker
2. comment
3. atomic configuration
4. number of channels, spin-orbit channel count, radial count, radial-grid
   parameters `a,b`, ionic valence `Zion`, and optional core parameters
5. positive radial grid
6. each angular channel `l=0..3` followed by tabulated `r*V_l`
7. radial core charge
8. radial valence charge
9. each `l`, atomic occupation, cutoff, and reduced radial wavefunction `u_l`

On input,

```text
V_l(r_i) = [r V_l(r_i)] / r_i
rho(r_i) = radial_charge(r_i) / [4 pi r_i²]
```

and `u_l=r R_l`. The tabulated tail obeys `r V_l -> -2 Zion`.

The local component is not encoded in POTRE. `Local_Component` is required per
species (`usrinputfile.F90:1806-1827`), and the user is responsible for avoiding
ghost states.

### Local ionic potential

After choosing `l_local`,

```text
V_local = V_l_local
Delta V_l = V_l - V_local
```

(`pseudo.f90:829-860`). `ionpot.f90:46-132` evaluates

```text
V_ion(r) = sum_atoms V_local_species(|r-R_atom|)
```

Normally PARSEC linearly interpolates `rV`, then divides by `r`. Beyond the
usable tabulated radius it uses the exact Rydberg Coulomb tail `-2 Zion/r`.
Spline interpolation is off by default.

The optional spline is the clamped Numerical Recipes `spline`/`splint`
algorithm in `spline.f90`, not a natural or not-a-knot cubic. If
`M=Expansion_Order/2`, `pseudo.f90` extends each positive POTRE radial table
with

```text
x_pad = [-M-1, -M, ..., -1, 0]
y_pad = [y(r_first), ..., y(r_first)]
```

and imposes `S'(x_pad_first)=S'(r_last)=0`. Thus the left endpoint condition
is applied at the far artificial knot, not at the first positive POTRE radius.
The local potential and NLCC core density use this padded spline down to the
origin and keep their usual tail/cutoff behavior. The initial valence-density
guess remains linearly interpolated even when `Cubic_Spline=true`.

### Initial atomic density and NLCC

If `Read_VCD=true`, `initchrg.f90` linearly interpolates the file valence
density. Otherwise `pseudo.f90:1035-1056` reconstructs

```text
rho_atom(r) = sum_l occupation_l * u_l(r)² / (4 pi r²)
```

The molecular starting density is the atom-centered sum, set to zero outside
each radial table, then normalized:

```text
Ne = sum_atoms Zion - net_charge
rho0 <- rho0 * Ne / [h³ sum_grid rho0]
```

For an NLCC potential, `corecd.f90` separately superposes the stored core
density. It is added only for XC:

```text
rho_XC = rho_valence + rho_core
```

The Hartree source remains the valence density.

### Kleinman-Bylander nonlocal term

For every `l != l_local`,

```text
beta_l(r) = Delta V_l(r) * u_l(r) / r
D_l = integral u_l(r)² Delta V_l(r) dr
beta_tilde_l = beta_l / sqrt(abs(D_l))
s_l = sign(D_l)
```

(`pseudo.f90:903-918,1096-1160`). Cartesian projectors are

```text
beta_alm(r) = beta_tilde_l(|r-Ra|) Y_lm(r-Ra)
```

using normalized real harmonics through `l=3`. They are stored only inside the
species cutoff sphere (`nonloc.F90`). `pseudo.f90:518-522` takes the largest
channel cutoff and advances it to the next radial-grid point;
`nonloc.F90:146-153` then clips that support to the penultimate usable radial
sample. To make atom-on-grid evaluation finite, `pseudo.f90:972-974` flattens
the first few radial projector samples and `nonloc.F90:539-540` clamps radii
below the first tabulated value before either linear or padded-spline
interpolation. The angular harmonics still use the actual displacement. With
default `Double_grid_order=1`, the discrete action is

```text
(V_NL psi)_i =
  sum_a,l,m s_l beta_alm(i)
  [h³ sum_j beta_alm(j) psi_j]
```

`Double_grid_order>1` adds Ono-Hirose subgrid averaging. That is separate from
the kinetic finite-difference order.

### Ion-ion energy

For an isolated system, `forceion.f90:39-96` uses

```text
E_II = 2 sum_(a<b) Za Zb / |Ra-Rb|
```

in Rydberg. There are no periodic compensation or self-energy terms in this
path.

## Hartree potential

### Equation and multipole boundary

`rho_hart.F90`, `hartset.F90`, and `hpotcg.F90` solve

```text
-∇² V_H = 8 pi rho_valence
V_H(r) = 2 integral rho(r') / |r-r'| dr'
```

For normalized complex spherical harmonics, the equivalent moments are

```text
Q_lm = integral rho(r) r^l conjugate(Y_lm(rhat)) dr
```

and an exterior value is

```text
V_H(R) =
  2 sum_l [4 pi/(2l+1)]
    sum_m Q_lm Y_lm(Rhat) / R^(l+1)
```

The default/effective maximum `l` is 9. Multipoles are centered at the domain
origin.

For every active row and every stencil neighbor outside the domain, `hartset`
moves the known boundary value to the right-hand side:

```text
b_i = 8 pi rho_i - sum_(j outside) A_ij V_boundary(j)
```

Thus the Poisson matrix is the same interior `-∇²` matrix used by the kinetic
term, but its exterior values are multipole Dirichlet values rather than zero.

### Linear solver

`hpotcg.F90` uses unpreconditioned conjugate gradients with:

- relative tolerance `1e-7`
- absolute tolerance `1e-13`
- test `norm(r_k) <= 1e-7*norm(r_0) + 1e-13`
- maximum 1600 matrix-vector products

The first initial guess is `-V_ion`; later calls reuse the previous `V_H`
(`rho_hart.F90:158-175`).

`Full_Hartree=true` directly sums the density contribution at boundary points.
It is an expensive debugging path and is not the normal multipole algorithm.

## CA/PZ local-density functional

`Correlation_Type=ca`, `pz`, or `lda` selects the same unpolarized branch in
`exc_nspn.f90:247-269`.

For `rho_bar = rho_valence + rho_core > 0`,

```text
rs = [3/(4 pi rho_bar)]^(1/3)
a0 = [4/(9 pi)]^(1/3)
vx = -2/(pi a0 rs)
epsilon_x = 3 vx / 4
```

For `rs >= 1`,

```text
g=-0.2846, b1=1.0529, b2=0.3334
epsilon_c = g / (1 + b1 sqrt(rs) + b2 rs)
vc = epsilon_c²/g *
     [1 + 7 b1 sqrt(rs)/6 + 4 b2 rs/3]
```

For `rs < 1`,

```text
c1=0.0622, c2=0.096, c3=0.004, c4=0.0232, c5=0.0192
epsilon_c = c1 ln(rs)-c2 + [c3 ln(rs)-c4] rs
vc = epsilon_c -
     [c1 + (c3 ln(rs)-c5) rs]/3
```

Then

```text
V_xc = vx + vc
E_xc = h³ sum rho_bar * (epsilon_x + epsilon_c)
```

Nonpositive-density points receive zero. For NLCC, `E_xc` uses total
valence-plus-core density, but total-energy integrals of `rho*V_xc` use valence
density only.

## Hamiltonian and eigensolvers

The scalar isolated Hamiltonian is

```text
H = -∇²_FD + V_ion + V_H + V_xc + V_NL
```

All solvers ultimately call the same block matvec in `matvecB.f90z`.

### Effective defaults

| Setting | PARSEC default |
|---|---:|
| Eigensolver | Chebyshev-Davidson |
| First Chebyshev degree | 20 |
| Later subspace-filter degree | 15 |
| Lower/upper half degree offset | 3 |
| Extra state buffer | 6 |
| Diagonalization tolerance | `1e-4` |
| ChebFF filter cycles | 2 |
| CHEBFF bound request | 10, clamped to 8 in non-`BETA` |
| Later subspace bound request | 5 |
| CHEBFF matvec block | `min(States_Num/2, 6)` |

The first solve follows the selected eigensolver. For the `chebff` path used by
the bundled H2 and benzene inputs, `eigval.F90` calls `chebff_diag` once.
CHEBFF creates `States_Num + Subspace_Buffer_Size` trial vectors with
`DLARNV(IDIST=2)`, estimates the spectrum with short Lanczos, and then performs
exactly `FF_MaxIter` filter/orthonormalize/Rayleigh--Ritz cycles. It does not
test Ritz residuals: after the fixed work, all requested pairs are labelled
approximately converged.

If `chebdav` is selected instead, the first solve uses the distinct
Chebyshev--Davidson lock/expand/restart algorithm in `chebdav.f90z`. ARPACK is
another explicit first-solve selection. These solver names are not aliases.

Later ChebDav/ChebFF SCF iterations switch to `subspace.f90z`: reuse saved
vectors, apply one Chebyshev filter, orthonormalize, and perform Rayleigh-Ritz.
This later path declares the working subspace available without an eigenpair
residual test.

### Normalized Chebyshev recurrence

For unwanted interval `[cutoff, upper]`,

```text
e = (upper-cutoff)/2
c = (upper+cutoff)/2
sigma1 = e/(lambda_reference-c)
q1 = sigma1/e * (H-cI) q0

sigma_(j+1) = 1 / (2/sigma1 - sigma_j)
q_(j+1) = sigma_(j+1) *
          [2/e (H-cI) q_j - sigma_j q_(j-1)]
```

The lower half of the vectors uses degree `polym-delta`; the upper half uses
`polym+delta` (default 12 and 18). The upper spectral bound comes from a short
Lanczos estimate. The cutoff starts from the largest saved Ritz value, clipped
to at least zero, and is nondecreasing.

In the audited non-`BETA` source, the recurrence variable `sigma` is initialized
once per filter call and is carried across memory blocks. Therefore changing
`Matvec_Blocksize` can change the floating-point filter trajectory. The native
Python translation preserves this source behavior explicitly.

After iteration five, when SRE is below `100*vconv`, PARSEC reduces the degree
by one per iteration down to 10 (`parsec.F90:1446-1454`).

ARPACK is also available with smallest-algebraic targeting and
`ncv=2*nev+5`. `diagla` and TRLan are rejected in this commit.

## Occupations and density

The default Fermi temperature is 80 K and

```text
kB = 6.33327186e-6 Ry/K
```

`flevel.f90` bisects the chemical potential for

```text
f_i = 1 / [exp((epsilon_i-mu)/(kB T)) + 1]
sum_i f_i = Ne/2
```

Each `f_i` lies in `[0,1]`; the factor two for spin degeneracy is applied
elsewhere. The bisection count tolerance is `1e-12`, maximum iterations 100,
and exponent cutoff 35.

For full-grid, unpolarized orbitals with Euclidean column norm one,
`newrho.F90` builds

```text
rho(r_j) = (2/h³) sum_i f_i |psi_i(j)|²
```

so `h³ sum_j rho_j = Ne`.

## SCF loop, residual, and mixing

Let `V_in` be the mixed total effective potential used to diagonalize and

```text
V_out = V_ion + V_H[rho_new] + V_xc[rho_new+rho_core].
```

One iteration is:

1. solve `H[V_in] psi_i = epsilon_i psi_i`
2. determine Fermi occupations
3. build new valence density
4. solve new Hartree potential
5. evaluate new CA-LDA potential and energy
6. form `V_out`
7. evaluate energies and potential residual
8. mix `V_in` and `V_out`
9. test convergence

PARSEC mixes the effective potential, not the density.

### Convergence

`getsre.f90` computes

```text
SRE_weighted =
 sqrt[h³/Ne * sum_i rho_i (V_out_i-V_in_i)²]

SRE_plain =
 sqrt[h³ * sum_i (V_out_i-V_in_i)²]
```

The weighted value is the default. Defaults are:

- `Max_Iter = 50`
- `Convergence_Criterion = 2e-4 Ry`
- approach threshold `=100*vconv`

Energy change and density change are not stopping criteria.

### Anderson potential mixing

Default settings are parameter `beta=0.30`, memory 4, and restart every 20
iterations (`anderson.f90`). With residual `f_n=V_out-V_in`, the first step is

```text
x_(n+1) = x_n + beta f_n.
```

For retained history, define `df_i=f_n-f_(n-i)` and solve

```text
B_ij = dot(df_i,df_j)
b_i  = dot(df_i,f_n)
B c = b
```

then

```text
xbar = x_n + sum_i c_i (x_(n-i)-x_n)
fbar = f_n + sum_i c_i (f_(n-i)-f_n)
x_(n+1) = xbar + beta fbar.
```

No Kerker preconditioner is applied in this default isolated path.

## Total energy

For unpolarized occupations,

```text
E_band = 2 sum_i f_i epsilon_i.
```

PARSEC saves the input Hxc potential that generated the orbitals:

```text
V_Hxc_in = V_in - V_ion.
```

Using the new density and output Hartree/XC terms, `totnrg.f90` evaluates

```text
E_electronic =
 E_band
 - integral rho V_Hxc_in
 + 1/2 integral rho V_H_out
 + E_xc[rho+rho_core]

E_total = E_electronic + E_II.
```

Local and nonlocal pseudopotential contributions are already in `E_band`.
Finite-temperature occupations do not add an entropy term, so this is an
internal energy rather than a Mermin free energy.

## Source behaviors to treat carefully

- `Ignore_Symmetry` also changes the grid shift.
- `Solver_Lpole` is read but reset to 9 later in this commit.
- dynamic diagonalization tolerance is effectively prevented from tightening
  by the current input/loop assignments.
- later Chebyshev subspace iterations do not test Ritz residuals.
- the Chebyshev Fortran recurrence mutates a scalar across vector blocks; a
  clean port should reset recurrence state for each block.
- Poisson nonconvergence is printed but not propagated as a fatal SCF error.
- `Full_Hartree` is a slow debug-oriented boundary path.
- an origin-centered multipole series at a box face is not guaranteed to
  converge when density at a farther corner is nonzero; a direct boundary sum
  is the safe non-spherical alternative.
- `fdo5scl` is only a nearest-neighbor smoothing preconditioner for the removed
  DIAGLA path; it is not the physical Laplacian.
