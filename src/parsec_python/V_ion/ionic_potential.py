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
single-point path: spin-orbit projectors, periodic replicas, point charges
and forces are not handled here; PARSEC's optional ``Double_grid_order > 1``
fine-grid projector averaging lives in :mod:`.double_grid`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Mapping, Sequence
import warnings

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator

from ..Grid import RealSpaceGrid
from ..Grid.cell import is_diagonal_lattice, lattice_plane_spacings
from ..Grid.pbc import uses_fortran_routines
from ..models import Atom, SpeciesPotential
from ..Pseudopotential import (
    ParsecPseudopotential,
    ParsecRadialSpline,
    apply_fourier_filter,
    read_fhipp_pseudopotential,
    read_martins_old_pseudopotential,
    read_martins_wang_pseudopotential,
    read_parsec_pseudopotential,
)
from ..Pseudopotential.radial_derivatives import valence_density_table

#: ``eps`` of ``corecd.f90``: a superposed core density whose most negative
#: value is above ``-CORE_DENSITY_TOLERANCE`` electrons/bohr**3 is set to zero
#: where negative; a more negative one stops the run (error 381).
CORE_DENSITY_TOLERANCE = 0.1


def load_pseudopotentials(
    specifications: Mapping[str, SpeciesPotential],
    *,
    xc_functional: str = "ca",
    check_correlation: bool = True,
) -> dict[str, ParsecPseudopotential]:
    """Read and validate one PARSEC pseudopotential per chemical species.

    Besides parsing each POTRE file, this routine checks the assumptions made
    by the present scalar solver:

    * the species' physical element agrees with the symbol stored in the file;
    * the user-selected local angular-momentum channel exists;
    * a species requesting ``SO_PSP`` has a relativistic POTRE file carrying
      the matching p/d spin-orbit channels (without ``SO_PSP`` only the scalar
      components of a relativistic file are used); and
    * the pseudopotential was generated for the selected XC family.  A file
      of another functional is refused for CA and PBE, but for a CA file in a
      PBE run; for that one and for the other functionals it is used with a
      ``UserWarning``, as ``pseudo.f90`` (lines 383-394; PARSEC 1.5
      ``io/pseudo.f90:482-492``) only warns.

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
    A hybrid functional reads the files of its semilocal family
    (:func:`hybrid_pseudopotential_labels`: PBE, for B3PW91 also PW91): a CA
    file is used with a warning, as in a PBE run, any other is refused.
    ``check_correlation=False`` reads the files without that check (the input
    reader makes it itself, to refuse by name before the run).
    A libxc functional (``libxc:...``) reads the files of its family: CA/LDA
    files for an LDA, PBE files for everything else
    (:func:`parsec_python.V_xc.libxc.libxc_pseudopotential_functional`); a
    file of another label is used with a warning.
    """
    from ..V_xc.libxc import is_libxc_functional

    libxc = is_libxc_functional(xc_functional)
    result: dict[str, ParsecPseudopotential] = {}
    hybrid_labels = hybrid_pseudopotential_labels(xc_functional.strip().lower())
    requested = pseudopotential_functional(xc_functional.strip().lower())
    if requested not in _REQUIRED_PSEUDOPOTENTIAL_LABELS and (
        requested not in _EXPECTED_PSEUDOPOTENTIAL_LABELS
    ):
        raise ValueError(f"unsupported XC functional {xc_functional!r}")
    for symbol, specification in specifications.items():
        physical_element = specification.element_symbol or symbol
        if specification.pseudopotential_format in {"martins_wang", "fhipp"}:
            if specification.spin_orbit:
                # usrinputfile.F90 error 135 for FHIPP; pseudo.f90 reads the
                # spin-orbit columns of a Martins-Wang file into the wrong
                # array index.
                raise ValueError(
                    f"species {symbol!r}: SO_PSP is not supported for a "
                    f"{specification.pseudopotential_format} pseudopotential"
                )
            if specification.pseudopotential_format == "martins_wang":
                potential = read_martins_wang_pseudopotential(
                    specification.path,
                    symbol=physical_element,
                    core_cutoff_radius=specification.core_cutoff_radius,
                )
                # The file names its local channel; the input names none.
                if potential.local_channel != specification.local_angular_momentum:
                    raise ValueError(
                        f"species {symbol!r}: {specification.path} names the local "
                        f"channel l={potential.local_channel}, not "
                        f"l={specification.local_angular_momentum}"
                    )
            else:
                potential = read_fhipp_pseudopotential(
                    specification.path,
                    symbol=physical_element,
                    number_of_channels=specification.number_of_channels,
                    electrons_per_orbital=specification.electrons_per_orbital,
                    core_cutoff_radius=specification.core_cutoff_radius,
                    nonlinear_core_correction=specification.nonlinear_core_correction,
                )
                potential.validate_local_channel(specification.local_angular_momentum)
            result[symbol] = potential
            continue
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
        if not check_correlation:
            pass
        elif hybrid_labels is not None:
            if not pseudopotential_matches_functional(potential, xc_functional.strip().lower()):
                label = potential.correlation.strip().lower()
                if label not in _CA_LABELS:
                    raise ValueError(hybrid_pseudopotential_refusal(
                        symbol, specification.path, label, xc_functional.strip().lower()
                    ))
                warnings.warn(
                    f"{specification.path} was generated for {potential.correlation!r}, not for "
                    f"{hybrid_pseudopotential_family(xc_functional.strip().lower())}, the "
                    f"pseudopotentials of the hybrid {xc_functional}; it is used, as a PBE run uses it",
                    UserWarning,
                    stacklevel=2,
                )
        elif not pseudopotential_matches_functional(potential, requested):
            if libxc:
                warnings.warn(
                    f"{specification.path} was generated for {potential.correlation!r}, not for "
                    f"{requested.upper()}, the pseudopotentials of the libxc functional {xc_functional}; "
                    "it is used",
                    UserWarning,
                    stacklevel=2,
                )
                result[symbol] = potential
                continue
            if requested in _REQUIRED_PSEUDOPOTENTIAL_LABELS and not (
                requested == "pbe" and potential.correlation.strip().lower() in _CA_LABELS
            ):
                raise ValueError(
                    f"{specification.path} was generated for {potential.correlation!r}, "
                    f"not the requested {requested.upper()} functional"
                )
            warnings.warn(
                f"{specification.path} was generated for "
                f"{potential.correlation!r}, not for the requested {requested} "
                "functional; PARSEC warns ('correlation potential does not "
                "match') and uses the file, and so does this run",
                UserWarning,
                stacklevel=2,
            )
        result[symbol] = potential
    return result


def cluster_centering_shift(
    atoms: Sequence[Atom], threshold: float = 0.1
) -> np.ndarray | None:
    """The translation :func:`center_cluster_geometry` subtracts, or ``None``.

    ``(R_min + R_max)/2`` of the nuclei where its length exceeds
    ``threshold`` (bohr), ``None`` where the atoms stay where they are.
    ``initial.f90`` moves the point charges of the input by the same vector.
    """
    positions = np.asarray([atom.position for atom in atoms], dtype=float)
    shift = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    if np.linalg.norm(shift) <= threshold:
        return None
    return shift


# The correlation labels a pseudopotential file must carry for the CA and
# PBE functionals: any other label is refused, but for a CA file in a PBE
# run, which pseudo.f90 (PARSEC 1.5, io/pseudo.f90:482-492) only warns about,
# as for every other pair: the twelve GGA inputs of 1.5's Tier_xc tests are
# such runs.
_REQUIRED_PSEUDOPOTENTIAL_LABELS = {
    "ca": frozenset({"ca", "pz", "lda"}),
    "pbe": frozenset({"pb", "pbe"}),
}
_CA_LABELS = _REQUIRED_PSEUDOPOTENTIAL_LABELS["ca"]
# For the other functionals pseudo.f90 (lines 383-394) only warns when the
# label of the file differs from the two-letter code of the functional.
_EXPECTED_PSEUDOPOTENTIAL_LABELS = {
    "pw92": frozenset({"pl", "pw92", "pwlda"}),
    "pw91": frozenset({"pw", "pw91", "pwgga"}),
    "xalpha": frozenset({"xa"}),
    "wigner": frozenset({"wi"}),
    "hedin_lundqvist": frozenset({"hl"}),
    "lb94": frozenset({"lb"}),
    "casida_salahub": frozenset({"cs"}),
}


def pseudopotential_functional(xc_functional: str) -> str:
    """The functional whose pseudopotentials a calculation reads.

    A hybrid functional (``parsec_python.SCF.hybrid``) reads PBE
    pseudopotentials, as Quantum ESPRESSO's hybrid runs usually do: there are
    no hybrid norm-conserving pseudopotentials in the libraries of this code
    (B3PW91 also takes PW91 files, :func:`hybrid_pseudopotential_labels`).
    A meta-GGA of this package (``parsec_python.V_xc.meta_gga``) reads PBE
    pseudopotentials too, as its plane-wave references do.  A libxc
    functional reads CA files (an LDA) or PBE files (the others).  Every
    other functional reads its own.
    """

    from ..models import HYBRID_XC_FUNCTIONALS, META_GGA_XC_FUNCTIONALS
    from ..V_xc.libxc import is_libxc_functional, libxc_pseudopotential_functional

    if is_libxc_functional(xc_functional):
        return libxc_pseudopotential_functional(xc_functional)
    if xc_functional in HYBRID_XC_FUNCTIONALS or xc_functional in META_GGA_XC_FUNCTIONALS:
        return "pbe"
    return xc_functional


#: The correlation labels of the files each hybrid reads beyond PBE's: its semilocal family.
_HYBRID_EXTRA_LABELS = {"b3pw91": frozenset({"pw", "pw91", "pwgga"})}


def hybrid_pseudopotential_labels(xc_functional: str) -> frozenset | None:
    """The correlation labels of the pseudopotentials a hybrid reads, ``None`` for another functional.

    PBE files (``pb``, ``pbe``) for every hybrid; B3PW91 also PW91 files
    (``pw``, ``pw91``, ``pwgga``), the semilocal parent of its correlation.
    A CA file is used with a warning (as in a PBE run); any other label is
    refused, by the input reader before the run.
    """

    from ..models import HYBRID_XC_FUNCTIONALS

    if xc_functional not in HYBRID_XC_FUNCTIONALS:
        return None
    return _REQUIRED_PSEUDOPOTENTIAL_LABELS["pbe"] | _HYBRID_EXTRA_LABELS.get(xc_functional, frozenset())


def hybrid_pseudopotential_family(xc_functional: str) -> str:
    """Name of the pseudopotentials a hybrid reads (``"PBE"`` or ``"PBE or PW91"``)."""

    return "PBE or PW91" if xc_functional in _HYBRID_EXTRA_LABELS else "PBE"


def hybrid_pseudopotential_refusal(symbol: str, path, label: str, xc_functional: str) -> str:
    """The message that refuses a file of another functional under a hybrid."""

    labels = ", ".join(sorted(hybrid_pseudopotential_labels(xc_functional) or ()))
    return (
        f"Atom_Type {symbol}: {path} was generated for {label!r}; a {xc_functional} run reads "
        f"{hybrid_pseudopotential_family(xc_functional)} pseudopotentials (labels {labels}; a CA "
        "file is used with a warning)"
    )


def pseudopotential_matches_functional(
    potential: ParsecPseudopotential, xc_functional: str
) -> bool:
    """Whether the correlation label of a POTRE file names ``xc_functional``.

    An old Martins file names no functional and matches every one.
    """

    labels = hybrid_pseudopotential_labels(xc_functional)
    if labels is None:
        xc_functional = pseudopotential_functional(xc_functional)
        labels = _REQUIRED_PSEUDOPOTENTIAL_LABELS.get(
            xc_functional, _EXPECTED_PSEUDOPOTENTIAL_LABELS.get(xc_functional)
        )
    if labels is None:
        raise ValueError(f"unsupported XC functional {xc_functional!r}")
    correlation = getattr(potential, "correlation", None)
    if not correlation:
        return True
    return correlation.strip().lower() in labels


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
    Every other field of an atom is kept: the initial moment of a
    non-collinear run (``Initial_NCL_Moment``) was lost here before
    2026-11-05, so that an off-centre cluster started without its moments
    (Au2+ 0.19 bohr off the origin, two SCF steps with FF_MaxIter 4: then
    |m| = 0, with the moments kept |m| = 2.35).
    """
    positions = np.asarray([atom.position for atom in atoms], dtype=float)
    shift = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    if np.linalg.norm(shift) <= threshold:
        return tuple(atoms)
    return tuple(
        replace(atom, position=np.asarray(atom.position, dtype=float) - shift) for atom in atoms
    )


def center_slab_geometry(
    atoms: Sequence[Atom], threshold: float = 0.1
) -> tuple[tuple[Atom, ...], float]:
    """PARSEC's re-centring of a slab along its confined direction, in bohr.

    Translation of ``initial.f90:201-228`` (``pbc%per == 2``): with ``z_min``
    and ``z_max`` the extreme z coordinates of all atoms, the shift is
    ``z_mov = (z_min + z_max)/2``; when ``|z_mov| > threshold`` every atom
    moves by ``-z_mov`` along z, so that the highest and the lowest atom are
    equally far from the mid-plane ``z = 0`` of the slab box.  The in-plane
    coordinates are kept.  Returns ``(atoms, z_mov)``, with the atoms
    unchanged and ``z_mov = 0.0`` below the threshold.

    PARSEC applies it in a first run only (``istart == 0``) and not to
    ``manual`` movements, and moves neither point charges nor charged sheets
    (a cluster's point charges move with its atoms, a slab's do not); the
    caller decides when it applies.  Wires are not re-centred.
    """

    if not atoms:
        return tuple(atoms), 0.0
    heights = [float(np.asarray(atom.position, dtype=float)[2]) for atom in atoms]
    shift = 0.5 * (min(heights) + max(heights))
    if abs(shift) <= threshold:
        return tuple(atoms), 0.0
    offset = np.array([0.0, 0.0, shift])
    return (
        tuple(
            replace(atom, position=np.asarray(atom.position, dtype=float) - offset)
            for atom in atoms
        ),
        shift,
    )


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
            direct=specification.old_interpolation,
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


#: ``|r V + 2 Z| <= COULOMB_ROUND_OFF_TOLERANCE x 2 Z``: the table is the Coulomb tail to round-off from there on (the
#: POTRE files), the rule of :func:`parsec_python.acceleration.V_ion.periodic_device.coulomb_radius` and of
#: :func:`_coulomb_to_round_off`, which must stay equal for the POTRE tables to keep their radii.
COULOMB_ROUND_OFF_TOLERANCE = 1.0e-15

#: The Coulomb tail of a table written with fewer digits (:func:`coulomb_tail_radius`): ``|r V + 2 Z|`` within this
#: multiple of its largest value over the outer half of the table, and never below ``_COULOMB_TAIL_PRECISION 2 Z``;
#: a table whose outer half deviates by more than ``_COULOMB_TAIL_NOISE_LIMIT 2 Z`` has no Coulomb tail.
_COULOMB_TAIL_NOISE_FACTOR = 10.0
_COULOMB_TAIL_PRECISION = 1.0e-10
_COULOMB_TAIL_NOISE_LIMIT = 1.0e-6


def coulomb_tail_radius(potential: ParsecPseudopotential, local_l: int) -> float | None:
    """Radius from which ``r V_loc`` is ``-2 Z`` to the precision of its file, or ``None`` (bohr).

    For the tables whose ``r V`` never reaches ``-2 Z`` to the last bit: the FHIPP files of FHI98PP carry
    ``|r V + 2 Z| / 2 Z`` of 1e-13 to 1e-11 out to the end of the table (406 files of ``src/pp_generation/NCPPs``),
    the MARTINS_OLD and MARTINS_WANG files of the tree 5e-10 and 5e-8; the POTRE files are ``-2 Z`` to round-off
    (4e-15) beyond their core and never need this.  With the noise floor ``f = max |r V + 2 Z|`` over the outer half
    of the table (``r >= rs(ns)/2``, beyond any pseudopotential core), the tail starts at the sample after the last
    one with ``|r V + 2 Z| > max(10 f, 1e-10 x 2 Z)``: from 1.3 bohr (H) to 14 bohr (Cs) for the FHIPP files of the
    library, 6.1 bohr for its silicon.  Beyond it a periodic image adds
    to the Ewald split of the local potential only ``-2 Z erfc(eta r)/r`` and the noise of the file (below
    ``1e-9/r`` Ry per image for the FHIPP files), and, for a ``Cubic_Spline`` species, the error of the spline of
    ``V`` between the samples of ``-2 Z/r``, which grows to 1e-5 Ry near the end of the table and must not be summed
    over the images.  ``None`` where the outer half deviates by more than ``1e-6 x 2 Z`` (no Coulomb tail, such as
    ``examples/h2_full_nonlocal/H_POTRE.DAT``: 4.6e-5 x 2 Z) or no sample settles: the callers then sum to the end
    of the table, the deviation being part of the potential, and :func:`warn_spline_without_coulomb_tail` warns
    that a ``Cubic_Spline`` species adds the spline error of that tail too.
    """

    radii = np.asarray(potential.radii, dtype=float)
    two_z = 2.0 * float(potential.ionic_charge)
    deviation = np.abs(radii * np.asarray(potential.channel_potentials[local_l], dtype=float) + two_z)
    scale = max(two_z, 1.0)
    floor = float(np.max(deviation[radii >= 0.5 * radii[-1]], initial=0.0))
    if floor > _COULOMB_TAIL_NOISE_LIMIT * scale:
        return None
    tolerance = max(_COULOMB_TAIL_NOISE_FACTOR * floor, _COULOMB_TAIL_PRECISION * scale)
    significant = np.flatnonzero(deviation > tolerance)
    if significant.size == 0:
        return float(radii[0])
    following = int(significant[-1]) + 1
    if following >= radii.size - 1:
        return None
    return float(radii[following])


def warn_spline_without_coulomb_tail(potential: ParsecPseudopotential, local_l: int, reach: float) -> None:
    """Warn where the periodic sum of a ``Cubic_Spline`` local potential reaches the end of its table.

    A table that is not ``-2 Z/r`` at its end (:func:`coulomb_tail_radius` is ``None``) is summed over the images out
    to ``rs(ns-1)``, where the cubic spline of ``V`` misses ``-2 Z/r`` by up to 1e-5 Ry: the error enters every image
    (for a silicon crystal 0.05 Ry in the mean potential).  The linear interpolation of ``r V`` has no such error.
    A table with a Coulomb tail is silent even where the erfc reach passes its end (beyond it the sums take
    ``-2 Z/r`` itself).
    """

    if reach < float(potential.interpolation_cutoff):
        return
    if _coulomb_to_round_off(potential, local_l) or coulomb_tail_radius(potential, local_l) is not None:
        return
    import warnings

    warnings.warn(
        f"{potential.symbol}: the local potential (l={local_l}) of {getattr(potential, 'source', 'its file')} is not "
        f"-2Z/r at the end of its table, so its periodic images are summed out to {reach:.1f} bohr, where the cubic "
        "spline of Cubic_Spline: .true. adds its interpolation error (up to 1e-5 Ry per image); Cubic_Spline: .false. "
        "interpolates r V linearly without it",
        UserWarning,
        stacklevel=3,
    )


def coulomb_onset_radius(potential: ParsecPseudopotential, local_l: int) -> float:
    """Radius beyond which the local potential table is ``-2 Z/r`` to ``1e-14`` Ry (2 to 7 bohr for the tables in use).

    Beyond it a periodic image of the atom adds, to the Ewald split of the local potential, only the short-ranged
    ``-2 Z erfc(eta r)/r``; the tables themselves run to 118 bohr or more.  A table whose ``r V`` is ``-2 Z`` only to
    the digits of its file (FHIPP, MARTINS_OLD, MARTINS_WANG; not to ``1e-15`` relative) meets ``1e-14`` Ry late
    (25 bohr for a FHIPP silicon) or never: it takes :func:`coulomb_tail_radius` where that is smaller.
    """

    radii = np.asarray(potential.radii, dtype=float)
    deviation = np.abs(np.asarray(potential.channel_potentials[local_l], dtype=float) + 2.0 * potential.ionic_charge / radii)
    significant = np.flatnonzero(deviation > _COULOMB_ONSET_TOLERANCE)
    if significant.size == 0:
        return float(radii[1])
    onset = float(min(radii[min(int(significant[-1]) + 1, radii.size - 1)], potential.interpolation_cutoff))
    if not _coulomb_to_round_off(potential, local_l):
        tail = coulomb_tail_radius(potential, local_l)
        if tail is not None:
            return min(onset, tail)
    return onset


def _coulomb_to_round_off(potential: ParsecPseudopotential, local_l: int) -> bool:
    """Whether ``|r V + 2 Z| <= 1e-15 x 2 Z`` from a radius below ``rs(ns-1)`` on (the POTRE files): the rule of
    :func:`parsec_python.acceleration.V_ion.periodic_device.coulomb_radius` settles inside the table."""

    radii = np.asarray(potential.radii, dtype=float)
    two_z = 2.0 * float(potential.ionic_charge)
    deviation = np.abs(radii * np.asarray(potential.channel_potentials[local_l], dtype=float) + two_z)
    deviating = np.flatnonzero(deviation > COULOMB_ROUND_OFF_TOLERANCE * max(two_z, 1.0))
    if deviating.size == 0:
        return True
    following = int(deviating[-1]) + 1
    return following < radii.size and float(radii[following]) < potential.interpolation_cutoff


def superpose_atomic_density(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray | None = None,
    *,
    core: bool = False,
    periodic_dimensions: int = 3,
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

    A builder installed with :func:`set_atomic_density_provider` takes a
    periodic cell first (importing :mod:`parsec_python.acceleration` installs
    one that returns this sum bit for bit).
    """
    if _atomic_density_provider is not None and lattice_vectors is not None:
        provided = _atomic_density_provider(
            grid, atoms, potentials, specifications, lattice_vectors, core
        )
        if provided is not None:
            return provided
    density = np.zeros(grid.size, dtype=float)
    for atom in atoms:
        specification = specifications[atom.symbol]
        potential = potentials[atom.symbol]

        # Species without an NLCC table contribute no frozen core density.
        if core and not potential.has_nonlinear_core_correction:
            continue

        position = np.asarray(atom.position, dtype=np.float64)
        if lattice_vectors is not None and periodic_dimensions == 2:
            # a slab of Andrew Grieder's slab_boundary: the images along the in-plane vectors, to the table's end
            support = potential.interpolation_cutoff
            translations = _periodic_image_translations(lattice_vectors, support, periodic_dimensions)
        elif lattice_vectors is not None and (
            not uses_fortran_routines(grid)
            # Two sums of the Fortran-derived routines take his too (the same fields to 2.4e-10 Ry):
            # Fortran's general (oblique) grid, whose readable sum to the end of the table took 34-300 s, and a
            # Cubic_Spline core density of a bulk cell (the spline of the whole table at every image: 137 s, Ge3)
            or (
                not getattr(grid, "is_confined_periodic", False)
                and (getattr(grid, "is_general_periodic", False) or (core and specification.use_spline))
            )
        ):
            # Andrew Grieder's periodic sum: the images of the atom out to where its density is below 1e-16
            support = atomic_density_support_radius(potential, specification, core=core)
            translations = _periodic_image_translations(lattice_vectors, support)
        else:
            # an isolated system (one centre), and the Fortran-derived routines: the cutoff of the table
            support = potential.interpolation_cutoff
            translations = _image_translations(grid, lattice_vectors, position, support)
        reconstruct_from_wavefunctions = not core and not specification.read_valence_density
        # A table interpolated linearly below: the density rebuilt from the
        # radial functions, or the one fourier_f.f90 filtered, which replaces
        # either valence density.
        tabulated = reconstruct_from_wavefunctions
        if not core and potential.filtered_valence_density is not None:
            tabulated = True
            radial_density = np.asarray(potential.filtered_valence_density, dtype=float)
        elif reconstruct_from_wavefunctions:
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

        for translation in translations:
            rows, _, radius = _support_finder(grid, position + translation, support)
            inside = radius < support
            if not np.any(inside):
                continue
            rows, radius = rows[inside], radius[inside]
            if tabulated:
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


def check_core_density(core_density: np.ndarray) -> np.ndarray:
    """Apply the sign check of ``corecd.f90:193-213`` to a core density.

    A superposed core density with negative values whose most negative one
    lies within ``CORE_DENSITY_TOLERANCE`` of zero is set to zero there (a
    new array); a more negative one raises ``ValueError`` (Fortran error
    381).  Without negative values the array itself is returned.
    """

    values = np.asarray(core_density)
    if values.size == 0:
        return values
    lowest = float(np.min(values))
    if lowest >= 0.0:
        return values
    if abs(lowest) < CORE_DENSITY_TOLERANCE:
        return np.maximum(values, 0.0)
    raise ValueError(
        "negative core-correction charge density: its most negative value is "
        f"{lowest:.6g} electrons/bohr**3 (PARSEC stops with error 381 below "
        f"-{CORE_DENSITY_TOLERANCE:g})"
    )


def apply_fourier_filters(
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    spacing: float,
) -> dict[str, ParsecPseudopotential]:
    """Fourier-filter the species that ask for it (``fourier_f.f90``).

    ``spacing`` is the grid spacing of the periodic grid in bohr.  Species
    without ``Alpha_Filter``/``Core_Filter`` keep their tables; see
    :func:`parsec_python.Pseudopotential.apply_fourier_filter`.
    """

    result = dict(potentials)
    for symbol, specification in specifications.items():
        if not specification.has_fourier_filter:
            continue
        potential = potentials[symbol]
        result[symbol] = apply_fourier_filter(
            potential,
            specification.local_angular_momentum,
            spacing=spacing,
            alpha=specification.fourier_filter_alpha,
            beta1=specification.fourier_filter_beta1,
            core_alpha=specification.fourier_filter_core,
            valence_density=valence_density_table(
                potential, read_from_file=specification.read_valence_density
            ),
        )
    return result


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


PROJECTOR_CENTER_ROUND_OFF = 1.0e-6
"""Bohr: a projector's grid point closer than this (and than the first radial knot) to its atom is the center,
the clamp of the nonlocal forces (``Forces.periodic.ROUND_OFF_DISTANCE``)."""


def real_spherical_harmonics(
    angular_momentum: int,
    relative_coordinates: np.ndarray,
    *,
    center_radius: float = 0.0,
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

    ``center_radius`` (bohr, default 0): the cosines are ``x/max(r, c)``, so a
    point within ``c`` of the center goes continuously to that limit.  The
    projectors pass the radius their radial table is clamped to, as the forces
    do (:data:`parsec_python.Forces.periodic.ROUND_OFF_DISTANCE`): a grid point
    on an atom up to round-off (``1e-16`` to ``1e-12`` bohr) otherwise takes the
    random direction of its round-off, which breaks the symmetry of the
    operator (diamond with atoms on the points of its grid: ``2e-4`` of
    ``V_NL`` under a two-fold axis).

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
    unit[nonzero] = xyz[nonzero] / np.maximum(radius[nonzero], float(center_radius))[:, None]
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


_atomic_density_provider = None


def set_atomic_density_provider(provider=None) -> None:
    """Install (or, with ``None``, remove) a builder of the superposed densities of a periodic cell.

    ``provider(grid, atoms, potentials, specifications, lattice_vectors, core)`` returns the
    density of :func:`superpose_atomic_density`, or ``None`` for an input it does not take (the
    sum of :func:`superpose_atomic_density` then runs).  Importing :mod:`parsec_python.acceleration`
    installs the host builder of ``parsec_python.acceleration.V_ion.periodic_setup`` (the same
    sum in blocks of images, bit for bit); the accelerated command line installs the device builder
    of ``parsec_python.acceleration.V_ion.periodic_device`` for the forces, relaxation and dynamics
    of a periodic cell.
    """
    global _atomic_density_provider
    _atomic_density_provider = provider


def _periodic_image_translations(
    lattice_vectors: np.ndarray, support_radius: float, dimensions: int = 3
) -> np.ndarray:
    """Enumerate periodic translations whose projector image can reach the cell.

    Unlike the local ionic potential's Coulomb tail, a KB projector's radial
    function is *exactly* zero beyond ``support_radius`` (see
    ``build_nonlocal_projectors``'s ``np.interp(..., right=0.0)``), so no
    Ewald-style real/reciprocal split is needed here -- periodic images are a
    plain, exact finite sum; an image whose support ball cannot reach any grid
    point contributes exactly zero, not approximately.

    For each axis of an orthorhombic cell (diagonal ``lattice_vectors``), an
    image ``n`` cells away is offset by at least ``(n-1)`` times that axis's
    side length from a point already inside the cell (every atom position is
    translated into ``[0, side_length)`` by the periodic preparation), so
    translations with ``n`` beyond ``ceil(support_radius/side_length) + 1``
    cannot reach the cell and are excluded.

    A general cell uses the same bound with the distance between lattice
    planes, ``2*pi/|b_i|``, in place of the side length: grid points and
    atoms have fractional coordinates in ``[0, 1)``, so an image ``n_i``
    planes away along ``b_i`` lies at least ``(|n_i|-1)`` plane distances
    from every grid point.  (Fortran's ``nonloc.F90``, ``initchrg.f90`` and
    ``corecd.f90`` use ``+-1`` replicas only; this bound includes them.)
    """

    if not is_diagonal_lattice(lattice_vectors) or np.any(np.diag(lattice_vectors) <= 0.0):
        vectors = np.asarray(lattice_vectors, dtype=float)
        spacing = lattice_plane_spacings(vectors)
        max_index = np.ceil(support_radius / spacing).astype(int) + 1
        max_index[dimensions:] = 0  # a slab repeats only along its in-plane lattice vectors
        ranges = [np.arange(-m, m + 1) for m in max_index]
        grid = np.meshgrid(*ranges, indexing="ij")
        integer_combinations = np.column_stack([axis.reshape(-1) for axis in grid])
        return integer_combinations @ vectors
    side_lengths = np.diag(lattice_vectors)
    max_index = np.ceil(support_radius / side_lengths).astype(int) + 1
    max_index[dimensions:] = 0  # a slab repeats only along its in-plane lattice vectors
    ranges = [np.arange(-m, m + 1) for m in max_index]
    grid = np.meshgrid(*ranges, indexing="ij")
    integer_combinations = np.column_stack([axis.reshape(-1) for axis in grid])
    return integer_combinations * side_lengths


def _image_translations(
    grid, lattice_vectors: np.ndarray | None, position: np.ndarray, radius: float
) -> np.ndarray:
    """The translations of an atom's images that a radial function can reach.

    No translation for an isolated grid, those of
    :func:`_periodic_image_translations` for a periodic cell, and for the grid
    of a wire or slab
    (:class:`~parsec_python.Grid.confined.ConfinedPeriodicGrid`) the
    translations along its periodic directions only, from the atom's own
    position (``lattice_vectors`` is then not used).
    """

    if getattr(grid, "is_confined_periodic", False):
        return grid.image_translations(position, radius)
    if lattice_vectors is None:
        return np.zeros((1, 3))  # No translations if 0D
    return _periodic_image_translations(lattice_vectors, radius)


def _fractional_extent(grid, lattice_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(inverse, lowest, highest, widths)``: the inverse lattice, the range of the fractional coordinates of the
    grid points along each lattice vector and ``|inverse[:, m]|``, the fractional width of one bohr along it."""

    inverse = np.linalg.inv(np.asarray(lattice_vectors, dtype=float))
    fractional = np.asarray(grid.coordinates) @ inverse
    return inverse, fractional.min(axis=0), fractional.max(axis=0), np.linalg.norm(inverse, axis=0)


def _reaching_translations(extent, position: np.ndarray, radius: float, translations: np.ndarray) -> np.ndarray:
    """The translations whose ball of ``radius`` around ``position + T`` can hold a grid point (bohr).

    A point ``p`` within ``radius`` of a centre ``c`` has ``|f_m(p) - f_m(c)| <= radius |inverse[:, m]|`` along every
    lattice vector (``f = x @ inverse``), so a centre whose interval misses the fractional range of the grid holds
    none: the support finder would return no rows for it.  The test keeps a margin of 1e-9 for round-off; the
    images it keeps are searched as before, in their order.
    """

    inverse, lowest, highest, widths = extent
    if len(translations) <= 1:
        return translations
    centres = (np.asarray(position, dtype=float) + translations) @ inverse
    margin = 1.0e-9 * (1.0 + np.abs(centres))
    half = radius * widths
    keep = np.all((centres + half + margin >= lowest) & (centres - half - margin <= highest), axis=1)
    return translations[keep]


def build_nonlocal_projectors(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray | None = None,
    k_point: np.ndarray | None = None,
    periodic_dimensions: int = 3,
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
    the active grid, matching PARSEC's default ``Double_grid_order=1``.  With
    ``grid.settings.double_grid_order > 1`` they are averaged over PARSEC's
    finer subgrid instead
    (:func:`parsec_python.V_ion.double_grid.build_double_grid_projectors`).

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
    if int(getattr(grid.settings, "double_grid_order", 1)) > 1:
        from .double_grid import build_double_grid_projectors

        return build_double_grid_projectors(
            grid, atoms, potentials, specifications, lattice_vectors, k_point
        )
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
    # the images of a bulk cell whose support ball can hold a grid point (_reaching_translations)
    extent = (
        _fractional_extent(grid, lattice_vectors)
        if lattice_vectors is not None and periodic_dimensions == 3 and not getattr(grid, "is_confined_periodic", False)
        else None
    )
    # one radial table (and spline) per species and channel, shared by its atoms and their images
    channels: dict[tuple[str, int], tuple[np.ndarray, float, ParsecRadialSpline | None]] = {}

    for atom_index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        local_l = specifications[atom.symbol].local_angular_momentum

        position = np.asarray(atom.position, dtype=np.float64)

        support_radius = _projector_support_radius(potential)
        if lattice_vectors is not None and periodic_dimensions == 2:
            # a slab of Andrew Grieder's slab_boundary: images along its in-plane vectors only
            translations = _periodic_image_translations(lattice_vectors, support_radius, periodic_dimensions)
        else:
            translations = _image_translations(grid, lattice_vectors, position, support_radius)
        if extent is not None:
            translations = _reaching_translations(extent, position, support_radius, translations)

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
            key = (atom.symbol, angular_momentum)
            if key not in channels:
                radial_grid, denominator_sign = potential.radial_projector(
                    angular_momentum, local_l
                )
                channels[key] = (
                    radial_grid,
                    denominator_sign,
                    ParsecRadialSpline.from_positive_grid(
                        potential.radii,
                        radial_grid,
                        grid.settings.stencil_half_width,
                    )
                    if specifications[atom.symbol].use_spline
                    else None,
                )
            radial_grid, denominator_sign, radial_spline = channels[key]
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
                if radial_spline is not None:
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
                # A grid point on the atom up to round-off takes the center's
                # zero-limit direction (the radial value is clamped there too).
                harmonics = real_spherical_harmonics(
                    angular_momentum,
                    relative_support,
                    center_radius=min(float(potential.radii[0]), PROJECTOR_CENTER_ROUND_OFF),
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
