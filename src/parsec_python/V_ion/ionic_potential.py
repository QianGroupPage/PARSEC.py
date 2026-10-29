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
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator

from ..Grid import RealSpaceGrid
from ..models import Atom, SpeciesPotential
from ..Pseudopotential import (
    ParsecPseudopotential,
    ParsecRadialSpline,
    read_martins_old_pseudopotential,
    read_parsec_pseudopotential,
)


def load_pseudopotentials(
    specifications: Mapping[str, SpeciesPotential],
    *,
    xc_functional: str = "ca",
) -> dict[str, ParsecPseudopotential]:
    """Read and validate one PARSEC pseudopotential per chemical species.

    Besides parsing each POTRE file, this routine checks the assumptions made
    by the present scalar solver:

    * the species' physical element agrees with the symbol stored in the file;
    * the user-selected local angular-momentum channel exists;
    * a species requesting ``SO_PSP`` has a relativistic POTRE file carrying
      the matching p/d spin-orbit channels (without ``SO_PSP`` only the scalar
      components of a relativistic file are used); and
    * the pseudopotential was generated for the selected XC family.

    A file's spin-orbit channels (if present) are always parsed into
    :attr:`~parsec_python.Pseudopotential.ParsecPseudopotential.spin_orbit_channel_potentials`
    regardless of ``SO_PSP``, matching PARSEC's own read behavior. Applying
    the resulting spin-orbit projectors in the Hamiltonian -- Fortran's
    perturbative or self-consistent SOC -- is a separate capability this
    scalar single-point solver does not yet implement; requesting
    ``SO_PSP=true`` here only validates that the file supports it.

    The returned objects retain radial channel potentials ``V_l(r)``, reduced
    radial pseudo-wavefunctions ``u_l(r)``, occupations, cutoff radii, atomic
    valence/core densities, and the ionic valence charge ``Z_ion``.
    Checking the correlation label validates how the file was generated; this
    function does not itself evaluate an exchange-correlation functional.
    """
    result: dict[str, ParsecPseudopotential] = {}
    requested = xc_functional.strip().lower()
    compatible = {
        "ca": {"ca", "pz", "lda"},
        "pbe": {"pb", "pbe"},
    }
    if requested not in compatible:
        raise ValueError(f"unsupported XC functional {xc_functional!r}")
    for symbol, specification in specifications.items():
        physical_element = specification.element_symbol or symbol
        if specification.pseudopotential_format == "martins_old":
            # An old Martins file names neither its element nor its
            # functional, so there is nothing to check them against, as in
            # pseudo.f90.  Its occupations and core radius come from the
            # input of the species.
            potential = read_martins_old_pseudopotential(
                specification.path,
                symbol=physical_element,
                number_of_channels=specification.number_of_channels,
                electrons_per_orbital=specification.electrons_per_orbital,
                core_cutoff_radius=specification.core_cutoff_radius,
            )
            potential.validate_local_channel(specification.local_angular_momentum)
            result[symbol] = potential
            continue
        # Files are loaded once per species, not once per atom.
        potential = read_parsec_pseudopotential(specification.path)
        if potential.symbol.lower() != physical_element.lower():
            raise ValueError(
                f"species {symbol!r} represents element {physical_element!r}, but "
                f"the POTRE file contains symbol {potential.symbol!r} "
                f"in {specification.path}"
            )
        potential.validate_local_channel(specification.local_angular_momentum)
        # A relativistic file also carries the minor (spin-orbit) components
        # of its p and d channels.  Without SO_PSP, pseudo.f90 reads only the
        # scalar components, which are what the scalar solvers use.
        if specification.spin_orbit and not potential.has_spin_orbit_channels:
            raise ValueError(
                f"species {symbol!r} requests SO_PSP, but {specification.path} "
                f"does not contain spin-orbit potentials (relativity="
                f"{potential.relativity!r})"
            )
        pp_xc = potential.correlation.strip().lower()
        if pp_xc not in compatible[requested]:
            raise ValueError(
                f"{specification.path} was generated for {potential.correlation!r}, "
                f"not the requested {requested.upper()} functional"
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


_DENSITY_SUPPORT_TOLERANCE = 1.0e-16  # electrons/bohr^3: the atomic density tables are below it beyond their support
_COULOMB_ONSET_TOLERANCE = 1.0e-14  # Ry: the local potential tables are -2 Z/r to this beyond their onset


def _valence_density_table(potential: ParsecPseudopotential, specification: SpeciesPotential) -> np.ndarray:
    """The spherical atomic valence density on the radial grid of the table, as ``superpose_atomic_density`` uses it."""

    if specification.read_valence_density:
        return np.asarray(potential.valence_density, dtype=float)
    # u_l is the reduced radial function: the normalized spherical orbital is u_l(r)Y_lm(Omega)/r.  Spherical
    # occupation averaging produces f_l*u_l(r)**2/(4*pi*r**2).
    radial_density = np.zeros_like(potential.radii)
    for angular_momentum, wavefunction in potential.radial_wavefunctions.items():
        radial_density += (
            potential.channel_occupations.get(angular_momentum, 0.0)
            * wavefunction
            * wavefunction
            / (4.0 * np.pi * potential.radii * potential.radii)
        )
    return radial_density


def atomic_density_support_radius(
    potential: ParsecPseudopotential, specification: SpeciesPotential, *, core: bool
) -> float:
    """Radius beyond which the atomic valence (or core) density of a table is below ``1e-16`` electrons/bohr^3.

    The tables run to 118 bohr or more, but the densities vanish within 12 bohr (core) to 35 bohr (valence): a sum
    over periodic images needs no atom farther from a point than that.  The next table point is included, so that
    the interpolation between the last significant one and the next stays inside; never more than the table's own
    cutoff.
    """

    table = np.asarray(potential.core_density if core else _valence_density_table(potential, specification), dtype=float)
    significant = np.flatnonzero(np.abs(table) > _DENSITY_SUPPORT_TOLERANCE)
    if significant.size == 0:
        return float(potential.radii[1])
    index = min(int(significant[-1]) + 1, potential.radii.size - 1)
    return float(min(potential.radii[index], potential.interpolation_cutoff))


def coulomb_onset_radius(potential: ParsecPseudopotential, local_l: int) -> float:
    """Radius beyond which the local potential table is ``-2 Z/r`` to ``1e-14`` Ry (2 to 7 bohr for the tables in use).

    Beyond it a periodic image of the atom adds, to the Ewald split of the local potential, only the short-ranged
    ``-2 Z erfc(eta r)/r``; the tables themselves run to 118 bohr or more.
    """

    radii = np.asarray(potential.radii, dtype=float)
    deviation = np.abs(np.asarray(potential.channel_potentials[local_l], dtype=float) + 2.0 * potential.ionic_charge / radii)
    significant = np.flatnonzero(deviation > _COULOMB_ONSET_TOLERANCE)
    if significant.size == 0:
        return float(radii[1])
    return float(min(radii[min(int(significant[-1]) + 1, radii.size - 1)], potential.interpolation_cutoff))


def superpose_atomic_density(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray | None = None,
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

        position = np.asarray(atom.position, dtype=np.float64)
        # A periodic cell sums images of the atom out to where its density is below 1e-16; an isolated system has
        # one center and keeps the cutoff of the table.
        support = (
            potential.interpolation_cutoff
            if lattice_vectors is None
            else atomic_density_support_radius(potential, specification, core=core)
        )
        translations = (
            np.zeros((1, 3))  # No translations if 0D
            if lattice_vectors is None
            else _periodic_image_translations(lattice_vectors, support)
        )
        reconstruct_from_wavefunctions = not core and not specification.read_valence_density
        if reconstruct_from_wavefunctions:
            radial_density = _valence_density_table(potential, specification)

        for translation in translations:
            rows, _, radius = _support_finder(grid, position + translation, support)
            inside = radius < support
            if not np.any(inside):
                continue
            rows, radius = rows[inside], radius[inside]
            if reconstruct_from_wavefunctions:
                density[rows] += np.interp(
                    radius,
                    potential.radii,
                    radial_density,
                    left=radial_density[0],
                )
            else:
                # PARSEC's initial valence-density path is linear even when
                # the optional spline is used for other radial quantities.
                density[rows] += potential.interpolate_density(
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
    if len(atoms) < 2:
        return 0.0
    positions = np.array(
        [atom.position for atom in atoms], dtype=np.float64
    ).reshape(len(atoms), 3)
    charges = np.array(
        [potentials[atom.symbol].ionic_charge for atom in atoms],
        dtype=np.float64,
    )
    # One vectorized row per atom replaces the O(N^2) Python pair loop, which
    # took seconds beyond a few thousand atoms.  ``cumsum`` accumulates
    # strictly left to right, so pairs enter the sum in the original order.
    energy = 0.0
    for left in range(len(atoms) - 1):
        delta = positions[left] - positions[left + 1 :]
        distance = np.sqrt(
            (delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1])
            + delta[:, 2] * delta[:, 2]
        )
        if np.any(distance <= 0):
            raise ValueError("two nuclei occupy the same position")
        terms = 2.0 * charges[left] * charges[left + 1 :] / distance
        energy = float(np.cumsum(np.concatenate(((energy,), terms)))[-1])
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
        """Apply ``B*diag(signs)*B^H`` to one vector or a vector block.

        The first multiplication computes all projector overlaps
        ``c_q = <beta_q|psi> = sum_r conj(beta_q[r])*psi[r]``.  After
        applying the denominator signs, the second multiplication
        accumulates ``sum_q s_q*beta_q*c_q``.  ``B^H`` (conjugate
        transpose, not plain transpose) matters only for a k-point Bloch
        projector matrix (:func:`build_nonlocal_projectors`'s ``k_point``
        argument), which is genuinely complex; the ordinary real-harmonic
        Gamma-point projectors satisfy ``B^H=B.T`` so this is unchanged for
        them.  The input dtype is preserved (not forced to float) so a
        complex spinor or Bloch wavefunction is not silently truncated.
        """
        vectors = np.asarray(vectors)
        # B^H @ vectors evaluates every <beta_q|psi> overlap at once.
        coefficients = self.projectors.conj().T @ vectors
        if vectors.ndim == 1:
            coefficients = self.signs * coefficients
        else:
            coefficients = self.signs[:, None] * coefficients
        return np.asarray(self.projectors @ coefficients)

    def as_linear_operator(self) -> "LinearOperator":
        """Expose matrix-free projector application to SciPy eigensolvers."""
        from scipy.sparse.linalg import LinearOperator

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

    A ``MARTINS_OLD`` file declares no cutoff: its support radius is the
    ``Core_Cutoff_Radius`` of the input, capped in the same way.  Both rules
    live in :attr:`ParsecPseudopotential.projector_support_radius`.
    """
    return potential.projector_support_radius


def dense_support(grid, center: np.ndarray, radius: float):
    """Grid points within ``radius`` of ``center``: ``(rows, relative, distances)``.

    ``relative = grid.coordinates[rows] - center``.  This is the readable definition: it measures the
    distance of every grid point.  ``center`` may lie outside the cell (a periodic image).
    """
    relative = grid.coordinates - center
    distances = np.linalg.norm(relative, axis=1)
    rows = np.flatnonzero(distances <= radius)
    return rows, relative[rows], distances[rows]


_support_finder = dense_support


def set_support_finder(finder=None) -> None:
    """Install a faster ``finder(grid, center, radius)`` with the result of :func:`dense_support`.

    ``None`` restores :func:`dense_support`.  The performance counterpart lives in
    ``parsec_python.acceleration.V_ion.periodic_setup`` and installs itself when the acceleration
    package is imported.
    """
    global _support_finder
    _support_finder = dense_support if finder is None else finder


def _periodic_image_translations(
    lattice_vectors: np.ndarray, support_radius: float
) -> np.ndarray:
    """Enumerate periodic translations whose projector image can reach the cell.

    Unlike the local ionic potential's Coulomb tail, a KB projector's radial
    function is *exactly* zero beyond ``support_radius`` (see
    ``build_nonlocal_projectors``'s ``np.interp(..., right=0.0)``), so no
    Ewald-style real/reciprocal split is needed here -- periodic images are a
    plain, exact finite sum; an image whose support ball cannot reach any grid
    point contributes exactly zero, not approximately.

    Every atom position lies inside the cell (validated by the caller), so in
    fractional coordinates an image ``n`` cells away along lattice vector ``i``
    is offset by at least ``|n_i|-1`` from any point of the cell, and the
    fractional coordinate along ``i`` of a point at distance ``R`` changes by
    at most ``R/d_i`` with ``d_i`` the distance between the lattice planes
    ``i`` (``1/|b_i|``, ``b`` the reciprocal vectors without ``2 pi``).
    Translations with ``|n_i|`` beyond ``ceil(R/d_i) + 1`` cannot reach the
    cell and are excluded.  For an orthorhombic cell ``d_i`` is the side length.
    """

    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    plane_spacing = 1.0 / np.linalg.norm(np.linalg.inv(lattice_vectors), axis=0)
    max_index = np.ceil(support_radius / plane_spacing).astype(int) + 1
    ranges = [np.arange(-m, m + 1) for m in max_index]
    grid = np.meshgrid(*ranges, indexing="ij")
    integer_combinations = np.column_stack([axis.reshape(-1) for axis in grid])
    return integer_combinations @ lattice_vectors


def build_nonlocal_projectors(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray | None = None,
    k_point: np.ndarray | None = None,
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

    ``lattice_vectors``, if given, sums each ``(atom, l, m)`` projector over
    every periodic image whose support ball can reach the grid (see
    :func:`_periodic_image_translations`) -- unlike the local ionic
    potential's long-ranged Coulomb tail, a KB projector's radial function is
    exactly zero beyond its support radius, so this is a plain, exact finite
    sum, not an Ewald-style split.  ``None`` (the default) reproduces the
    original isolated single-copy behavior exactly.

    ``k_point``, if given (requires ``lattice_vectors``), builds the
    projector for the periodic Bloch factor ``u_k`` (``psi = e^{ik.r} u_k``,
    the representation the k-point kinetic operator acts in): each image
    ``T`` is weighted by the position-dependent phase ``exp(-i*k.(r-T))``,
    which is ``e^{-ik.r} * e^{ik.T}``, and the returned operator's
    ``projectors`` become complex.  ``None`` (the default) is the Gamma-point case, where every
    phase is exactly 1 and the projectors stay real.
    """
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    signs: list[float] = []
    labels: list[tuple[int, int, int]] = []
    column = 0
    complex_valued = k_point is not None
    value_dtype = complex if complex_valued else float
    if complex_valued and lattice_vectors is None:
        raise ValueError("k_point requires lattice_vectors")
    k_point = None if k_point is None else np.asarray(k_point, dtype=float)

    # Convert a continuous projector beta(r_i) into the Euclidean grid vector
    # b_i=sqrt(dV_grid)*beta(r_i).
    sqrt_dv = np.sqrt(grid.volume_element)

    for atom_index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        local_l = specifications[atom.symbol].local_angular_momentum

        position = np.asarray(atom.position, dtype=np.float64)

        support_radius = _projector_support_radius(potential)
        translations = (
            np.zeros((1, 3))  # No translations if 0D
            if lattice_vectors is None
            else _periodic_image_translations(lattice_vectors, support_radius)
        )

        # KB projectors are localized.  Restrict all following interpolation
        # and harmonic work to grid points within their radial support.
        images: list[tuple[np.ndarray, np.ndarray, np.ndarray, complex]] = []
        for translation in translations:
            support_rows, relative_support, radius_support = _support_finder(
                grid, position + translation, support_radius
            )
            if support_rows.size:
                # Bloch factor of this image's projector in the periodic-u
                # picture (psi = e^{ik.r} u, the picture the k-point kinetic
                # term -nabla^2 - 2ik.grad + k^2 acts in):
                #   e^{-ik.r} * sum_T e^{ik.T} beta(r - R - T)
                # i.e. a *position-dependent* phase e^{-ik.(r-T)} per image,
                # not just the constant e^{ik.T}.  They coincide at k = 0.
                phase = (
                    1.0
                    if k_point is None
                    else np.exp(
                        -1j * ((grid.coordinates[support_rows] - translation) @ k_point)
                    )
                )
                images.append((support_rows, relative_support, radius_support, phase))
        if not images:
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
            # One column per real harmonic (2*l+1 of them), shared by every
            # image below: a periodic projector is one function per (atom,
            # l, m), not one per image, so images accumulate into the same
            # column rather than adding separate ones.  COO->CSC sums
            # duplicate (row, column) entries, so an image that happens to
            # overlap another image's rows near a small cell's edge is
            # handled automatically.
            channel_columns = list(range(column, column + 2 * angular_momentum + 1))
            column += len(channel_columns)
            for harmonic_index in range(len(channel_columns)):
                signs.append(denominator_sign)
                labels.append((atom_index, angular_momentum, harmonic_index))

            for support_rows, relative_support, radius_support, phase in images:
                if specifications[atom.symbol].use_spline:
                    radial_spline = ParsecRadialSpline.from_positive_grid(
                        potential.radii,
                        radial_grid,
                        grid.settings.stencil_half_width,
                    )
                    # ``nonloc.F90`` clamps the radial interpolation coordinate
                    # to the first positive POTRE radius.  Angular harmonics
                    # still use the actual atom-image-grid displacement below.
                    interpolation_radius = np.maximum(
                        radius_support, potential.radii[0]
                    )
                    radial = radial_spline(interpolation_radius)
                else:
                    radial = np.interp(
                        radius_support,
                        potential.radii,
                        radial_grid,
                        left=radial_grid[0],
                        right=0.0,
                    )

                # The same normalized radial function is paired with every one
                # of the 2*l+1 real harmonics in this angular-momentum channel.
                harmonics = real_spherical_harmonics(
                    angular_momentum, relative_support
                )
                for harmonic_index, target_column in enumerate(channel_columns):
                    projector = phase * (sqrt_dv * radial * harmonics[:, harmonic_index])

                    # Drop only values at numerical zero so the stored columns
                    # remain sparse without changing physically relevant entries.
                    keep = np.abs(projector) > 1.0e-16
                    if not np.any(keep):
                        continue
                    rows.append(support_rows[keep])
                    columns.append(
                        np.full(np.count_nonzero(keep), target_column, dtype=np.int64)
                    )
                    values.append(projector[keep].astype(value_dtype, copy=False))

    # CSC is natural here because projectors are stored and contracted by
    # column.  An all-local pseudopotential legitimately produces zero columns.
    if column == 0:
        matrix = sp.csc_matrix((grid.size, 0), dtype=value_dtype)
    else:
        matrix = sp.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(grid.size, column),
            dtype=value_dtype,
        ).tocsc()
    return NonlocalProjectorOperator(
        projectors=matrix,
        signs=np.asarray(signs, dtype=float),
        labels=tuple(labels),
    )
