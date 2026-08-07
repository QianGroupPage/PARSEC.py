"""Local and separable nonlocal ionic terms for isolated calculations.

For a norm-conserving pseudopotential, the electron-ion operator is split as

``V_ion = V_local + V_nonlocal``.

The local contribution is diagonal on the real-space grid:

``V_local(r) = sum_a V_local,a(|r-R_a|)``.

Outside the radial pseudopotential table, each atomic contribution has the
Rydberg Coulomb tail ``-2*Z_a/|r-R_a|``.  The separable Kleinman-Bylander (KB)
part is represented as a sum of low-rank projectors,

``V_nonlocal = sum_(a,l,m) sign(D_a,l) |beta_a,l,m><beta_a,l,m|``.

This module also builds the superposed atomic starting density, the nonlinear
core density used by exchange-correlation, and the isolated ion-ion energy.
Lengths are in bohr, potentials and energies are in Rydberg, and volume
densities are in electrons/bohr**3.  Its scope is the scalar, nonperiodic
single-point path: spin-orbit projectors, periodic replicas, point charges,
forces, and PARSEC's optional ``Double_grid_order > 1`` fine-grid projector
averaging are not handled here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from ..Grid import RealSpaceGrid
from ..models import Atom, SpeciesPotential
from ..Pseudopotential import (
    ParsecPseudopotential,
    ParsecRadialSpline,
    read_parsec_pseudopotential,
)


def load_pseudopotentials(
    specifications: Mapping[str, SpeciesPotential],
) -> dict[str, ParsecPseudopotential]:
    """Read and validate one PARSEC pseudopotential per chemical species.

    Besides parsing each POTRE file, this routine checks the assumptions made
    by the present scalar CA-LDA solver:

    * the dictionary key agrees with the element symbol stored in the file;
    * the user-selected local angular-momentum channel exists;
    * no spin-orbit channels are present; and
    * the pseudopotential was generated for the CA/PZ LDA family.

    The returned objects retain radial channel potentials ``V_l(r)``, reduced
    radial pseudo-wavefunctions ``u_l(r)``, occupations, cutoff radii, atomic
    valence/core densities, and the ionic valence charge ``Z_ion``.
    Checking the correlation label validates how the file was generated; this
    function does not itself evaluate an exchange-correlation functional.
    """
    result: dict[str, ParsecPseudopotential] = {}
    for symbol, specification in specifications.items():
        # Files are loaded once per species, not once per atom.
        potential = read_parsec_pseudopotential(specification.path)
        if potential.symbol.lower() != symbol.lower():
            raise ValueError(
                f"species key {symbol!r} does not match symbol {potential.symbol!r} "
                f"in {specification.path}"
            )
        potential.validate_local_channel(specification.local_angular_momentum)
        if potential.number_of_spin_orbit_channels:
            raise ValueError(
                f"{specification.path} contains spin-orbit channels; "
                "the scalar single-point solver does not support them"
            )
        if potential.correlation.strip().lower() not in {"ca", "pz", "lda"}:
            raise ValueError(
                f"{specification.path} was generated for {potential.correlation!r}, "
                "not the requested CA/PZ LDA"
            )
        result[symbol] = potential
    return result


def center_cluster_geometry(atoms: Sequence[Atom], threshold: float = 0.1) -> tuple[Atom, ...]:
    """Apply PARSEC's optional bounding-box recentering in bohr.

    If ``R_min`` and ``R_max`` are the componentwise extrema of all nuclear
    coordinates, the proposed translation is the bounding-box midpoint

    ``shift = (R_min + R_max)/2``.

    When ``|shift| > threshold``, every atom is translated as
    ``R_a <- R_a-shift``.  Otherwise the original coordinates are retained.
    This is not a center-of-mass or center-of-charge transformation; it only
    centers the smallest axis-aligned box containing the nuclei.

    The operation is separate from grid construction so callers can retain
    absolute coordinates when desired.  It performs only the translation;
    the caller decides whether recentering is enabled for a calculation.
    """
    positions = np.asarray([atom.position for atom in atoms], dtype=float)
    shift = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    if np.linalg.norm(shift) <= threshold:
        return tuple(atoms)
    return tuple(Atom(atom.symbol, atom.position - shift) for atom in atoms)


def build_local_ionic_potential(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
) -> np.ndarray:
    """Evaluate the total local electron-ion potential on the active grid.

    For atom ``a`` at ``R_a`` with selected local channel ``l_local(a)``,

    ``V_ion,local(r_i) = sum_a V_a,l_local(|r_i-R_a|)``.

    The radial evaluator interpolates the tabulated POTRE channel close to the
    atom and uses ``-2*Z_a/r`` beyond the table, as required in Rydberg units.
    ``use_spline=False`` follows PARSEC's default interpolation of ``r*V(r)``;
    the optional spline path interpolates ``V(r)`` directly.

    The returned one-dimensional array is the diagonal local potential in
    Rydberg; no sparse diagonal matrix is materialized here.  Because nuclei
    and POTRE data are fixed in a single-point calculation, this contribution
    is built once and remains constant throughout SCF.
    """
    total = np.zeros(grid.size, dtype=float)
    for atom in atoms:
        # An isolated calculation uses the direct Euclidean atom-grid
        # displacement; there is no periodic minimum-image convention.
        radius = np.linalg.norm(grid.coordinates - atom.position, axis=1)
        specification = specifications[atom.symbol]
        total += potentials[atom.symbol].local_potential(
            radius,
            specification.local_angular_momentum,
            use_spline=specification.use_spline,
            spline_padding_width=grid.settings.stencil_half_width,
        )
    return total


def superpose_atomic_density(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    *,
    core: bool = False,
) -> np.ndarray:
    """Superpose spherical atomic densities on the molecular grid.

    For the ordinary initial valence density,

    ``rho_0(r_i) = sum_a rho_a,valence(|r_i-R_a|)``.

    This is only an SCF starting guess and is normalized separately by
    :func:`normalize_density`.  With ``core=True`` the same operation builds
    the frozen nonlinear-core-correction density

    ``rho_core(r_i) = sum_a rho_a,core(|r_i-R_a|)``.

    The core density is supplied to the exchange-correlation functional; it
    is not part of the valence electron count or Hartree source.

    When a stored valence-density table is not requested, the spherical
    density is reconstructed from reduced radial functions ``u_l(r)`` and
    channel occupations ``f_l``:

    ``rho_a(r) = sum_l f_l*u_l(r)**2 / (4*pi*r**2)``.

    Stored valence densities follow PARSEC's linear ``initchrg`` interpolation
    and all atomic-density contributions are zero beyond their radial-table
    cutoff.  Core density may use the optional spline setting.
    """
    density = np.zeros(grid.size, dtype=float)
    for atom in atoms:
        specification = specifications[atom.symbol]
        potential = potentials[atom.symbol]

        # Species without an NLCC table contribute no frozen core density.
        if core and not potential.has_nonlinear_core_correction:
            continue
        radius = np.linalg.norm(grid.coordinates - atom.position, axis=1)
        if not core and not specification.read_valence_density:
            # u_l is the reduced radial function: the normalized spherical
            # orbital is u_l(r)Y_lm(Omega)/r.  Spherical occupation averaging
            # produces f_l*u_l(r)**2/(4*pi*r**2).
            radial_density = np.zeros_like(potential.radii)
            for angular_momentum, wavefunction in potential.radial_wavefunctions.items():
                radial_density += (
                    potential.channel_occupations.get(angular_momentum, 0.0)
                    * wavefunction
                    * wavefunction
                    / (4.0 * np.pi * potential.radii * potential.radii)
                )
            contribution = np.zeros_like(radius)
            inside = radius < potential.interpolation_cutoff
            contribution[inside] = np.interp(
                radius[inside],
                potential.radii,
                radial_density,
                left=radial_density[0],
            )
            density += contribution
        else:
            # PARSEC's initial valence-density path is linear even when the
            # optional spline is used for other radial quantities.
            density += potential.interpolate_density(
                radius,
                core=core,
                # ``initchrg.f90`` always uses linear interpolation.
                use_spline=specification.use_spline if core else False,
                spline_padding_width=grid.settings.stencil_half_width,
            )
    return density


def normalize_density(
    density: np.ndarray,
    grid: RealSpaceGrid,
    electron_count: float,
) -> np.ndarray:
    """Scale an initial volume density to the requested valence charge.

    On the uniform active grid,

    ``N_raw = integral rho(r) d^3r ~= DeltaV*sum_i rho_i``.

    This routine returns

    ``rho_normalized = rho * electron_count/N_raw``,

    so ``grid.integrate(rho_normalized) == electron_count`` up to roundoff.
    It returns a copy and does not modify the caller's array.
    """
    density = np.asarray(density, dtype=float).copy()
    integral = grid.integrate(density)
    if integral <= 0:
        raise ValueError("initial atomic density has a nonpositive integral")
    density *= electron_count / integral
    return density


def ionic_charge(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
) -> float:
    """Return ``sum_a Z_ion,a``, the total pseudopotential valence charge.

    For a neutral calculation this is also the requested valence-electron
    count.  Core electrons already absorbed into the pseudopotential are not
    included.
    """
    return float(sum(potentials[atom.symbol].ionic_charge for atom in atoms))


def ion_ion_energy(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
) -> float:
    """Return the isolated pairwise ion-ion repulsion in Rydberg.

    For pseudopotential ionic charges ``Z_a``, the Hartree-unit Coulomb energy
    is ``sum_(a<b) Z_a*Z_b/R_ab``.  Conversion to Rydberg gives

    ``E_II = 2*sum_(a<b) Z_a*Z_b/|R_a-R_b|``.

    This is the nonperiodic molecular expression; no Ewald or minimum-image
    sum is performed.
    """
    energy = 0.0
    for left, atom_i in enumerate(atoms):
        charge_i = potentials[atom_i.symbol].ionic_charge
        for atom_j in atoms[left + 1 :]:
            distance = float(np.linalg.norm(atom_i.position - atom_j.position))
            if distance <= 0:
                raise ValueError("two nuclei occupy the same position")
            charge_j = potentials[atom_j.symbol].ionic_charge
            energy += 2.0 * charge_i * charge_j / distance
    return float(energy)


def real_spherical_harmonics(
    angular_momentum: int,
    relative_coordinates: np.ndarray,
) -> np.ndarray:
    """Evaluate a real orthonormal spherical-harmonic basis through ``l=3``.

    For every relative coordinate ``r-R_a``, the returned row contains the
    ``2*l+1`` real angular functions ``Y_l,mu(Omega)`` satisfying

    ``integral Y_l,mu(Omega)*Y_l,nu(Omega) dOmega = delta_mu,nu``.

    The implementation uses normalized Cartesian polynomials in the direction
    cosines ``x/r``, ``y/r``, and ``z/r``.  At the exact atomic center the
    direction is undefined; the code uses zero direction cosines.  This is
    the stable zero-limit convention for regular ``l>0`` channels, while
    ``Y_00`` is direction independent.

    Columns span the complete channel.  Orthogonal rotations, permutations,
    or sign changes within that channel leave
    ``sum_mu |beta_l,mu><beta_l,mu|`` unchanged, which is why the precise real
    harmonic ordering does not alter the scalar KB operator.
    """
    xyz = np.asarray(relative_coordinates, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("relative_coordinates must have shape (n, 3)")
    radius = np.linalg.norm(xyz, axis=1)
    unit = np.zeros_like(xyz)
    nonzero = radius > 0
    unit[nonzero] = xyz[nonzero] / radius[nonzero, None]
    x, y, z = unit.T

    # l=0: s-like constant harmonic Y_00 = 1/sqrt(4*pi).
    if angular_momentum == 0:
        return np.full((xyz.shape[0], 1), 0.28209479177387814)
    # l=1: p-like real basis proportional to (x, y, z)/r.
    if angular_momentum == 1:
        return 0.4886025119029199 * np.column_stack((x, y, z))
    # l=2: five d-like real harmonics.
    if angular_momentum == 2:
        c = 1.0925484305920792
        return np.column_stack(
            (
                c * x * y,
                c * y * z,
                c * x * z,
                0.31539156525252005 * (3.0 * z * z - 1.0),
                0.5 * c * (x * x - y * y),
            )
        )
    # l=3: seven f-like real harmonics.
    if angular_momentum == 3:
        return np.column_stack(
            (
                0.5900435899266435 * y * (3.0 * x * x - y * y),
                2.890611442640554 * x * y * z,
                0.4570457994644658 * y * (5.0 * z * z - 1.0),
                0.3731763325901154 * z * (5.0 * z * z - 3.0),
                0.4570457994644658 * x * (5.0 * z * z - 1.0),
                1.445305721320277 * z * (x * x - y * y),
                0.5900435899266435 * x * (x * x - 3.0 * y * y),
            )
        )
    raise ValueError("PARSEC scalar projectors support angular momenta l=0 through l=3")


@dataclass(frozen=True)
class NonlocalProjectorOperator:
    """Low-rank discrete Kleinman-Bylander nonlocal operator.

    Let column ``q`` of ``B`` be one normalized discrete projector
    ``beta_a,l,m`` and let ``s_q = sign(D_a,l)``.  The operator is

    ``V_NL = B*diag(s)*B.T
           = sum_q s_q |beta_q><beta_q|``.

    ``projectors`` has shape ``(number_of_grid_points, number_of_projectors)``.
    ``labels[q]`` is ``(atom_index, angular_momentum, harmonic_index)`` and is
    retained for inspection/debugging.  Grid wavefunctions and projector
    columns use PARSEC's Euclidean normalization, with physical volume factors
    absorbed into the vectors.
    """

    projectors: sp.csc_matrix
    signs: np.ndarray
    labels: tuple[tuple[int, int, int], ...]

    @property
    def shape(self) -> tuple[int, int]:
        """Square grid-space shape of the represented nonlocal operator."""
        return (self.projectors.shape[0], self.projectors.shape[0])

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """Apply ``B*diag(signs)*B.T`` to one vector or a vector block.

        The first multiplication computes all projector overlaps
        ``c_q = <beta_q|psi>``.  After applying the denominator signs, the
        second multiplication accumulates ``sum_q s_q*beta_q*c_q``.
        """
        vectors = np.asarray(vectors, dtype=float)
        # B.T @ vectors evaluates every <beta_q|psi> overlap at once.
        coefficients = self.projectors.T @ vectors
        if vectors.ndim == 1:
            coefficients = self.signs * coefficients
        else:
            coefficients = self.signs[:, None] * coefficients
        return np.asarray(self.projectors @ coefficients)

    def as_linear_operator(self) -> LinearOperator:
        """Expose matrix-free projector application to SciPy eigensolvers."""
        return LinearOperator(
            self.shape,
            matvec=self.apply,
            matmat=self.apply,
            rmatvec=self.apply,
            dtype=float,
        )

    def as_sparse(self) -> sp.csr_matrix:
        """Materialize ``B*diag(signs)*B.T`` for tests or small systems.

        Production Hamiltonian applications use :meth:`apply`, avoiding a
        potentially much denser grid-by-grid matrix.
        """
        if self.projectors.shape[1] == 0:
            return sp.csr_matrix(self.shape)
        return (self.projectors @ sp.diags(self.signs) @ self.projectors.T).tocsr()


def _projector_support_radius(potential: ParsecPseudopotential) -> float:
    """Choose the grid-aligned outer support radius for KB projectors.

    The requested physical support is the largest channel cutoff in the
    POTRE file.  The returned value is the first tabulated radius strictly
    beyond that cutoff, capped at the penultimate radial point.  Keeping this
    extra interpolation shell avoids cutting a projector between radial-grid
    samples while still excluding its zero tail.  This matches PARSEC's
    default ``Double_grid_order=1`` support construction; the fallback used
    when a file declares no channel cutoffs is a defensive Python path.
    """
    requested = max(potential.channel_cutoffs.values(), default=potential.radii[0])
    next_index = int(np.searchsorted(potential.radii, requested, side="right"))
    next_index = min(next_index, potential.radii.size - 2)
    return float(potential.radii[next_index])


def build_nonlocal_projectors(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
) -> NonlocalProjectorOperator:
    """Construct the separable nonlocal pseudopotential on the active grid.

    For a nonlocal channel ``l != l_local``, define

    ``DeltaV_l(r) = V_l(r)-V_local(r)``,

    with reduced radial pseudo-wavefunction ``u_l(r)``.  The KB denominator is

    ``D_l = integral u_l(r)**2 * DeltaV_l(r) dr``.

    A normalized continuous projector is

    ``beta_a,l,m(r) = [DeltaV_l(r_a)*u_l(r_a)/r_a]
                       * Y_l,m(Omega_a) / sqrt(abs(D_l))``,

    where ``r_a=|r-R_a|``.  The channel contribution is then

    ``V_NL,a,l = sign(D_l) * sum_m |beta_a,l,m><beta_a,l,m|``.

    Eigensolver wavefunctions are Euclidean-normalized grid vectors
    ``q_i=sqrt(dV_grid)*psi(r_i)``.  Each stored projector therefore includes
    the same ``sqrt(dV_grid)`` factor, making ``B.T@q`` the real-space
    quadrature approximation to the continuous overlap.

    One sparse CSC column is created for every ``(atom,l,m)``.  The selected
    local channel is skipped because it is already included by
    :func:`build_local_ionic_potential`.  Projectors are sampled directly on
    the active grid, matching PARSEC's default ``Double_grid_order=1``.  Its
    optional finer ``Double_grid_order > 1`` subgrid averaging is not part of
    this implementation.
    """
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    signs: list[float] = []
    labels: list[tuple[int, int, int]] = []
    column = 0

    # Convert a continuous projector beta(r_i) into the Euclidean grid vector
    # b_i=sqrt(dV_grid)*beta(r_i).
    sqrt_dv = np.sqrt(grid.volume_element)

    for atom_index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        local_l = specifications[atom.symbol].local_angular_momentum
        relative = grid.coordinates - atom.position
        radius = np.linalg.norm(relative, axis=1)

        # KB projectors are localized.  Restrict all following interpolation
        # and harmonic work to grid points within their radial support.
        support = radius <= _projector_support_radius(potential)
        support_rows = np.flatnonzero(support)
        if support_rows.size == 0:
            continue

        for angular_momentum in sorted(potential.radial_wavefunctions):
            # V_local is handled as a diagonal potential and has no separable
            # projector of its own.
            if angular_momentum == local_l:
                continue

            # radial_projector returns DeltaV_l*u_l/r/sqrt(|D_l|) together
            # with sign(D_l), which is applied by NonlocalProjectorOperator.
            radial_grid, denominator_sign = potential.radial_projector(
                angular_momentum, local_l
            )
            if specifications[atom.symbol].use_spline:
                radial_spline = ParsecRadialSpline.from_positive_grid(
                    potential.radii,
                    radial_grid,
                    grid.settings.stencil_half_width,
                )
                # ``nonloc.F90`` clamps the radial interpolation coordinate
                # to the first positive POTRE radius.  Angular harmonics still
                # use the actual atom-grid displacement below.
                interpolation_radius = np.maximum(
                    radius[support], potential.radii[0]
                )
                radial = radial_spline(interpolation_radius)
            else:
                radial = np.interp(
                    radius[support],
                    potential.radii,
                    radial_grid,
                    left=radial_grid[0],
                    right=0.0,
                )

            # The same normalized radial function is paired with every one of
            # the 2*l+1 real harmonics in this angular-momentum channel.
            harmonics = real_spherical_harmonics(
                angular_momentum, relative[support]
            )
            for harmonic_index in range(harmonics.shape[1]):
                projector = sqrt_dv * radial * harmonics[:, harmonic_index]

                # Drop only values at numerical zero so the stored columns
                # remain sparse without changing physically relevant entries.
                keep = np.abs(projector) > 1.0e-16
                rows.append(support_rows[keep])
                columns.append(np.full(np.count_nonzero(keep), column, dtype=np.int64))
                values.append(projector[keep])
                signs.append(denominator_sign)
                labels.append((atom_index, angular_momentum, harmonic_index))
                column += 1

    # CSC is natural here because projectors are stored and contracted by
    # column.  An all-local pseudopotential legitimately produces zero columns.
    if column == 0:
        matrix = sp.csc_matrix((grid.size, 0), dtype=float)
    else:
        matrix = sp.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(grid.size, column),
        ).tocsc()
    return NonlocalProjectorOperator(
        projectors=matrix,
        signs=np.asarray(signs, dtype=float),
        labels=tuple(labels),
    )
