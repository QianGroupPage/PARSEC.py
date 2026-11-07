"""Public data models for the modular PARSEC-style single-point solver.

All lengths are in bohr, densities are in electrons/bohr**3, and energies and
potentials are in Rydberg unless a field explicitly says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import re
from typing import Literal, Mapping, Sequence

import numpy as np


DomainShape = Literal["sphere", "box", "ellipsoid", "cylinder"]
# ``none`` is PARSEC's TEST mode (``Eigensolver: none``): setup only.
EigensolverName = Literal["chebff", "chebdav", "arpack", "diagla", "none"]
# Eigensolvers whose first solve PARSEC always follows by the later-SCF
# Chebyshev SUBSPACE filter (usrinputfile.F90 forces Chebyshev_Filtering).
ALWAYS_FILTERING_EIGENSOLVERS = frozenset({"chebff", "chebdav"})
# Maximum_matvec default of usrinputfile.F90.
DEFAULT_MAXIMUM_MATVEC = 30000
HartreeBoundaryMethod = Literal["auto", "multipole", "direct"]
HartreeAtomicTail = Literal["auto", "on", "off"]
# The largest multipole order of the Hartree boundary every builder accepts.
# The recurrences are sound beyond it; the normalization prefactor of Y_lm
# becomes subnormal at l = m = 86 and would silently drop terms.
MAXIMUM_MULTIPOLE_ORDER = 60
# ``Correlation_Type`` of PARSEC: ca, pl (pw92), xa (xalpha), wi (wigner),
# hl (hedin_lundqvist), pb (pbe), pw (pw91), lb (lb94), cs (casida_salahub);
# see ``parsec_python.V_xc.selection``.
XCFunctional = Literal[
    "ca",
    "pbe",
    "pw92",
    "xalpha",
    "wigner",
    "hedin_lundqvist",
    "pw91",
    "lb94",
    "casida_salahub",
    "hf",
    "pbe0",
    "pbeh",
    "hse06",
    "hse03",
    "b3lyp",
    "b3lyp5",
    "b3pw91",
    "lc-wpbe",
    "cam-b3lyp",
]
# Self-consistent hybrid functionals (generalized Kohn-Sham with a nonlocal
# exchange operator, parsec_python.SCF.hybrid): the labels of the registry
# parsec_python.V_xc.hybrid.HYBRID_FUNCTIONALS; no Fortran PARSEC label.
HYBRID_XC_FUNCTIONALS = frozenset(
    ("hf", "pbe0", "pbeh", "hse06", "hse03", "b3lyp", "b3lyp5", "b3pw91", "lc-wpbe", "cam-b3lyp")
)
XC_FUNCTIONAL_NAMES = frozenset(
    (
        "ca",
        "pbe",
        "pw92",
        "xalpha",
        "wigner",
        "hedin_lundqvist",
        "pw91",
        "lb94",
        "casida_salahub",
    )
) | HYBRID_XC_FUNCTIONALS
# Functionals without a spin-polarized form (none in Fortran's exc_spn.f90).
UNPOLARIZED_ONLY_XC_FUNCTIONALS = frozenset(("xalpha", "wigner", "hedin_lundqvist"))
# ``sad``, ``file``, or a density model of ``MLDensity.registry`` (built in:
# ``charge3net``, ``scdp``, ``custom``).
InitialDensityMethod = str
DensityUnits = Literal[
    "auto",
    "e_per_bohr3",
    "e_per_angstrom3",
    "electrons_per_voxel",
]
DensityNegativePolicy = Literal["clip", "error", "allow"]
# ``Initial_Spin_Polarization`` default of ``usrinputfile.F90``.
DEFAULT_INITIAL_SPIN_POLARIZATION = 0.1


@dataclass(frozen=True)
class Atom:
    """One atom in Cartesian bohr coordinates."""

    symbol: str
    position: np.ndarray | Sequence[float]
    initial_moment: np.ndarray | Sequence[float] | None = None
    """Direction of this atom's initial magnetic moment for a non-collinear
    calculation (Fortran's per-atom ``Initial_NCL_Moment``).  Only the
    direction matters: it is normalized here, and the initial magnitude comes
    from the species' ``initial_spin_polarization``.  ``None`` (or the zero
    vector) seeds no moment on this atom."""

    def __post_init__(self) -> None:
        symbol = self.symbol.strip()
        position = np.asarray(self.position, dtype=float)
        if not symbol:
            raise ValueError("atom symbol cannot be empty")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("atom position must contain three finite Cartesian coordinates")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "position", position)
        if self.initial_moment is not None:
            moment = np.asarray(self.initial_moment, dtype=float)
            if moment.shape != (3,) or not np.all(np.isfinite(moment)):
                raise ValueError("initial_moment must be three finite numbers")
            norm = float(np.linalg.norm(moment))
            object.__setattr__(
                self, "initial_moment", None if norm == 0.0 else moment / norm
            )


@dataclass(frozen=True)
class PointCharge:
    """A classical point charge of a confined system (``Add_Point_Charges``).

    ``charge`` in e (positive for a positive charge, as ``Pt_Chg``) and
    ``position`` in Cartesian bohr.  It acts on the electrons through the
    local potential and on the nuclei through a Coulomb energy and force
    (:mod:`parsec_python.V_ion.external`); it is held fixed.
    """

    charge: float
    position: np.ndarray | Sequence[float]

    def __post_init__(self) -> None:
        charge = float(self.charge)
        position = np.asarray(self.position, dtype=float)
        if not np.isfinite(charge):
            raise ValueError("a point charge must be finite")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(
                "a point charge position must contain three finite Cartesian coordinates"
            )
        object.__setattr__(self, "charge", charge)
        object.__setattr__(self, "position", position)


@dataclass(frozen=True)
class ChargedSheet:
    """A uniformly charged plane ``z = height`` of a slab (``add_charged_sheets``).

    ``charge`` is the charge of the sheet per surface cell in e (positive
    for a positive charge, as ``sheet_charge``) and ``height`` its ``z`` in
    bohr (the third component of a ``Sheet_Coord`` line, F
    ``usrinputfile.F90:2222-2299``).  It adds ``4*pi*(q/A)*|z - height|``
    (Ry) to the local potential of the electrons and an electrostatic energy
    and force on the ions (:mod:`parsec_python.V_ion.wire_slab`).
    """

    charge: float
    height: float

    def __post_init__(self) -> None:
        charge = float(self.charge)
        height = float(self.height)
        if not (np.isfinite(charge) and np.isfinite(height)):
            raise ValueError("a charged sheet needs a finite charge and height")
        object.__setattr__(self, "charge", charge)
        object.__setattr__(self, "height", height)


@dataclass(frozen=True)
class SpeciesPotential:
    """Location and input-only choices for one pseudopotential species.

    The mapping key used by :class:`SinglePointInput` is a *species label*,
    not necessarily a bare element.  For example, normal carbon and a carbon
    1s-hole potential can coexist as ``"C"`` and ``"C-1s"``.  In that case
    ``element_symbol`` records that both POTRE files describe the chemical
    element C.  If omitted it is inferred from the leading element token.

    ``atomic_energy_correction`` is the optional all-electron minus
    pseudo-atom reference energy in Rydberg for this exact PP.  It is a
    constant reporting correction and never enters the Hamiltonian or SCF.
    Such a correction is required before comparing total energies obtained
    with ordinary and core-hole pseudopotentials having different valence
    charges.

    ``spin_orbit`` mirrors PARSEC's per-species ``SO_PSP`` flag and requires
    the POTRE file to be a relativistic (``irel='rel'``) potential carrying
    explicit p/d spin-orbit channels (see
    :attr:`~parsec_python.Pseudopotential.ParsecPseudopotential.has_spin_orbit_channels`).
    It switches on the spin-orbit term of the spin-orbit drivers (Fortran's
    perturbative or self-consistent SOC); without it only the scalar components
    of a relativistic file are used.

    ``pseudopotential_format`` is ``"martins_new"`` (a ``*_POTRE.DAT`` file)
    or ``"martins_old"`` (``*_POTR.DAT``).  An old file holds neither
    occupations nor a cutoff radius, and PARSEC reads them from the input:
    ``number_of_channels`` (``Potential_Num``), ``electrons_per_orbital``
    (the ``Electron_Per_Orbital`` block: s, p, d, f; their sum is the ionic
    charge) and ``core_cutoff_radius`` (``Core_Cutoff_Radius`` in bohr, the
    radius of the nonlocal projectors).  All three are required for an old
    file and refused for a new one, which holds its own.

    ``initial_spin_polarization`` is PARSEC's ``Initial_Spin_Polarization``
    of the species, read only by a spin-polarized calculation: the atomic
    starting density of the species is multiplied by one plus it for the
    up channel and by one minus it for the down channel
    (``initchrg.f90``).  The default is that of ``usrinputfile.F90``.

    ``atomic_radius`` is PARSEC's ``Atomic_Radius`` in bohr: the radius of
    the sphere around each atom of the species within which a non-collinear
    calculation sums the magnetization into an atomic moment (``nclmom`` of
    ``bxc.F90``).  It is a diagnostic and changes nothing of the calculation.

    ``hubbard_u`` and ``hubbard_j`` are PARSEC's per-species ``ldaplusu_u``
    and ``ldaplusu_j`` in Ry: the on-site Coulomb and exchange parameters
    of the LDA+U term of the d channel (``usrinputfile.F90``,
    :mod:`parsec_python.Hamiltonian.hubbard`).  Both zero, the default,
    is no LDA+U term.

    ``"martins_wang"`` (``*_POTRW.DAT``) takes everything from its file but
    the projector radius: ``core_cutoff_radius`` is required, the other two
    are refused, and ``local_angular_momentum`` must be the local channel
    the file names.  ``"fhipp"`` (``*_FHIPP.DAT``) needs all three, like
    ``"martins_old"``, and reads its core density only with
    ``nonlinear_core_correction`` (``Nonlinear_Core_Correction``, refused
    for the other formats, which say it in the file).

    ``old_interpolation`` is ``Old_Interpolation_Format`` of the input, set
    for every species of a cluster: the local potential and its radial
    derivative are interpolated directly instead of as ``r*V`` and
    ``r**2*dV/dr`` (``ionpot.f90``, ``forcloc.f90``).

    ``fourier_filter_alpha``, ``fourier_filter_beta1`` and
    ``fourier_filter_core`` are ``Alpha_Filter``, ``Beta1_Filter`` and
    ``Core_Filter`` of a species of a periodic system (``None`` for the last
    is ``Alpha_Filter``); zero filters nothing
    (:mod:`parsec_python.Pseudopotential.fourier_filter`).
    """

    path: str | Path
    local_angular_momentum: int
    read_valence_density: bool = False
    use_spline: bool = False
    element_symbol: str | None = None
    atomic_energy_correction: float = 0.0
    spin_orbit: bool = False
    pseudopotential_format: str = "martins_new"
    number_of_channels: int | None = None
    electrons_per_orbital: tuple[float, ...] | None = None
    core_cutoff_radius: float | None = None
    initial_spin_polarization: float = DEFAULT_INITIAL_SPIN_POLARIZATION
    atomic_radius: float = 0.0
    hubbard_u: float = 0.0
    hubbard_j: float = 0.0
    nonlinear_core_correction: bool = False
    old_interpolation: bool = False
    fourier_filter_alpha: float = 0.0
    fourier_filter_beta1: float = 0.0
    fourier_filter_core: float | None = None

    @property
    def has_hubbard(self) -> bool:
        """Whether the species carries an LDA+U term (``U`` or ``J`` nonzero)."""

        return self.hubbard_u != 0.0 or self.hubbard_j != 0.0

    @property
    def has_fourier_filter(self) -> bool:
        core = (
            self.fourier_filter_alpha
            if self.fourier_filter_core is None
            else self.fourier_filter_core
        )
        return self.fourier_filter_alpha > 0.0 or core > 0.0

    def __repr__(self) -> str:
        # Fields added later print nothing at their defaults, so the text of
        # every calculation without them stays what it was before they existed
        # (it is hashed by the cache keys of the accelerated driver).
        hidden = {
            "spin_orbit": not self.spin_orbit,
            "initial_spin_polarization": (
                self.initial_spin_polarization == DEFAULT_INITIAL_SPIN_POLARIZATION
            ),
            "atomic_radius": self.atomic_radius == 0.0,
            "hubbard_u": self.hubbard_u == 0.0,
            "hubbard_j": self.hubbard_j == 0.0,
            "nonlinear_core_correction": not self.nonlinear_core_correction,
            "old_interpolation": not self.old_interpolation,
            "fourier_filter_alpha": self.fourier_filter_alpha == 0.0,
            "fourier_filter_beta1": self.fourier_filter_beta1 == 0.0,
            "fourier_filter_core": self.fourier_filter_core is None,
        }
        shown = [
            f"{item.name}={getattr(self, item.name)!r}"
            for item in fields(self)
            if not hidden.get(item.name, False)
        ]
        return f"{type(self).__qualname__}({', '.join(shown)})"

    def __post_init__(self) -> None:
        local_l = int(self.local_angular_momentum)
        if local_l != self.local_angular_momentum or not 0 <= local_l <= 3:
            raise ValueError("local angular momentum must be 0 (s) through 3 (f)")
        self._check_format_input(local_l)
        element = self.element_symbol
        if element is None:
            # Species labels are supplied by the surrounding mapping, so a
            # missing element is resolved by the input translator.  Direct
            # API users must provide it when their key is not the element.
            element = None
        elif not re.fullmatch(r"[A-Z][a-z]?", element.strip()):
            raise ValueError("element_symbol must be a conventional element symbol")
        correction = float(self.atomic_energy_correction)
        if not np.isfinite(correction):
            raise ValueError("atomic_energy_correction must be finite")
        polarization = float(self.initial_spin_polarization)
        if not np.isfinite(polarization):
            raise ValueError("initial_spin_polarization must be finite")
        if abs(polarization) > 1.0:
            raise ValueError(
                "initial_spin_polarization must lie between -1 and 1: the "
                "starting spin densities are (1 + p) and (1 - p) times the "
                f"atomic density, got {polarization:g}"
            )
        object.__setattr__(self, "initial_spin_polarization", polarization)
        radius = float(self.atomic_radius)
        if not np.isfinite(radius):
            raise ValueError("atomic_radius must be finite")
        object.__setattr__(self, "atomic_radius", radius)

        filters = [float(self.fourier_filter_alpha), float(self.fourier_filter_beta1)]
        if self.fourier_filter_core is not None:
            filters.append(float(self.fourier_filter_core))
        if not all(np.isfinite(value) and value >= 0.0 for value in filters):
            raise ValueError("the Fourier filter parameters must be nonnegative")
        object.__setattr__(self, "fourier_filter_alpha", filters[0])
        object.__setattr__(self, "fourier_filter_beta1", filters[1])
        if self.fourier_filter_core is not None:
            object.__setattr__(self, "fourier_filter_core", filters[2])
        object.__setattr__(self, "nonlinear_core_correction", bool(self.nonlinear_core_correction))
        object.__setattr__(self, "old_interpolation", bool(self.old_interpolation))
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "local_angular_momentum", local_l)
        object.__setattr__(
            self,
            "element_symbol",
            None if element is None else element.strip(),
        )
        object.__setattr__(self, "atomic_energy_correction", correction)
        for name in ("hubbard_u", "hubbard_j"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    def _check_format_input(self, local_l: int) -> None:
        """Validate what the input adds to a pseudopotential file."""

        pseudo_format = str(self.pseudopotential_format).strip().lower()
        if pseudo_format not in {"martins_new", "martins_old", "martins_wang", "fhipp"}:
            raise ValueError(
                "pseudopotential_format must be 'martins_new', 'martins_old', "
                "'martins_wang' or 'fhipp'"
            )
        object.__setattr__(self, "pseudopotential_format", pseudo_format)
        if self.nonlinear_core_correction and pseudo_format != "fhipp":
            raise ValueError(
                "nonlinear_core_correction belongs to an fhipp pseudopotential; "
                "the other formats say in their file whether the core is corrected"
            )
        given = (
            self.number_of_channels,
            self.electrons_per_orbital,
            self.core_cutoff_radius,
        )
        if pseudo_format == "martins_new":
            if any(value is not None for value in given):
                raise ValueError(
                    "number_of_channels, electrons_per_orbital and "
                    "core_cutoff_radius belong to a martins_old pseudopotential"
                )
            return
        if pseudo_format == "martins_wang":
            if self.number_of_channels is not None or self.electrons_per_orbital is not None:
                raise ValueError(
                    "a martins_wang pseudopotential takes its channels and "
                    "occupations from its file; give only core_cutoff_radius"
                )
            if self.core_cutoff_radius is None:
                raise ValueError(
                    "a martins_wang pseudopotential needs core_cutoff_radius"
                )
            radius = float(self.core_cutoff_radius)
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError("core_cutoff_radius must be positive")
            object.__setattr__(self, "core_cutoff_radius", radius)
            return
        if any(value is None for value in given):
            raise ValueError(
                f"a {pseudo_format} pseudopotential needs number_of_channels, "
                "electrons_per_orbital and core_cutoff_radius"
            )
        channels = int(self.number_of_channels)
        if channels != self.number_of_channels or not 1 <= channels <= 4:
            raise ValueError("number_of_channels must be 1 to 4")
        if local_l >= channels:
            raise ValueError(
                "the local angular momentum lies beyond the number of channels"
            )
        electrons = tuple(float(value) for value in self.electrons_per_orbital)
        if len(electrons) < channels or not np.all(np.isfinite(electrons)):
            raise ValueError(
                "electrons_per_orbital needs a finite occupation for every channel"
            )
        radius = float(self.core_cutoff_radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("core_cutoff_radius must be positive")
        object.__setattr__(self, "number_of_channels", channels)
        # PARSEC reads one occupation per channel and ignores the rest.
        object.__setattr__(self, "electrons_per_orbital", electrons[:channels])
        object.__setattr__(self, "core_cutoff_radius", radius)


def _repr_without_defaults(settings, defaults: Mapping[str, object]) -> str:
    """The dataclass ``repr`` without the fields that hold the given defaults."""

    shown = [
        f"{item.name}={getattr(settings, item.name)!r}"
        for item in fields(settings)
        if not (item.name in defaults and getattr(settings, item.name) == defaults[item.name])
    ]
    return f"{type(settings).__qualname__}({', '.join(shown)})"


def _check_double_grid_order(settings) -> None:
    order = int(settings.double_grid_order)
    if order != settings.double_grid_order or order < 1:
        # usrinputfile.F90 error 113.
        raise ValueError("double_grid_order must be a positive integer")
    object.__setattr__(settings, "double_grid_order", order)


@dataclass(frozen=True)
class GridSettings:
    """PARSEC cluster-grid settings.

    ``expansion_order`` is the user-facing PARSEC order.  It must be even;
    half that many neighbors are used on each side of a point.

    ``double_grid_order`` is PARSEC's ``Double_Grid_Order`` ``N``: with
    ``N > 1`` the nonlocal projectors are averaged over ``(2N-1)**3``
    points of a grid ``N`` times finer around every grid point
    (Ono and Hirose, :mod:`parsec_python.V_ion.double_grid`).

    ``domain_shape`` is PARSEC's ``Cluster_Domain_Shape`` (``grid_partition.f90``
    ``outside_domain``): ``sphere`` of ``radius``, ``box`` of the full side
    lengths ``box_lengths``, ``ellipsoid`` with the semi-axes
    ``domain_parameters = (r_x, r_y, r_z)`` and ``cylinder`` with
    ``domain_parameters = (radius, length, axis)`` (axis 1, 2 or 3 for x, y,
    z; ``|coordinate| <= length/2`` along it), all in bohr.

    ``foldblock_side_length`` is PARSEC 1.5's ``Foldblock_Side_Length``
    (``grid_partition.f90:525-607``): with 2 the domain is filled up to whole
    blocks of 2 x 2 x 2 points starting at even indices (a block enters when
    any of its points lies inside the shape); 1 is the shape alone.
    """

    spacing: float
    radius: float
    expansion_order: int = 12
    shift: tuple[float, float, float] = (0.5, 0.5, 0.5)
    domain_shape: DomainShape = "sphere"
    box_lengths: tuple[float, float, float] | None = None
    double_grid_order: int = 1
    domain_parameters: tuple[float, float, float] | None = None
    foldblock_side_length: int = 1

    def __repr__(self) -> str:
        # The double grid, the parameters of an ellipsoid or a cylinder and
        # the fold blocks print nothing at their defaults, so the text of every
        # calculation without them stays what it was (cache keys hash it).
        return _repr_without_defaults(
            self,
            {"double_grid_order": 1, "domain_parameters": None, "foldblock_side_length": 1},
        )

    def __post_init__(self) -> None:
        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("grid spacing must be positive")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("domain radius must be positive")
        _check_double_grid_order(self)
        order = int(self.expansion_order)
        if order != self.expansion_order:
            raise ValueError("PARSEC expansion_order must be an integer")
        if order < 2 or order > 20 or order % 2:
            raise ValueError("PARSEC expansion_order must be an even integer from 2 to 20")
        if len(self.shift) != 3 or not np.all(np.isfinite(self.shift)):
            raise ValueError("grid shift must have three finite entries")
        if self.domain_shape not in {"sphere", "box", "ellipsoid", "cylinder"}:
            raise ValueError(
                "cluster domains are sphere, box, ellipsoid or cylinder"
            )
        if self.domain_shape in {"ellipsoid", "cylinder"}:
            parameters = self.domain_parameters
            if parameters is None or len(parameters) != 3:
                raise ValueError(
                    f"a {self.domain_shape} domain needs three domain_parameters"
                )
            values = tuple(float(value) for value in parameters)
            if self.domain_shape == "ellipsoid" and (
                not np.all(np.isfinite(values)) or min(values) <= 0.0
            ):
                raise ValueError("the radii of an ellipsoid must be positive")
            if self.domain_shape == "cylinder" and (
                not np.all(np.isfinite(values[:2]))
                or min(values[:2]) <= 0.0
                or values[2] not in (1.0, 2.0, 3.0)
            ):
                raise ValueError(
                    "a cylinder needs a positive radius and length and an axis 1, 2 or 3"
                )
            object.__setattr__(self, "domain_parameters", values)
        if self.domain_shape == "box":
            lengths = self.box_lengths
            if lengths is None:
                lengths = (2.0 * self.radius,) * 3
                object.__setattr__(self, "box_lengths", lengths)
            lengths_array = np.asarray(lengths, dtype=float)
            if (
                len(lengths) != 3
                or not np.all(np.isfinite(lengths_array))
                or np.any(lengths_array <= 0)
            ):
                raise ValueError("box_lengths must contain three positive lengths")
            object.__setattr__(
                self, "box_lengths", tuple(float(value) for value in lengths_array)
            )
        object.__setattr__(self, "expansion_order", order)
        object.__setattr__(self, "shift", tuple(float(x) for x in self.shift))
        side = int(self.foldblock_side_length)
        if side != self.foldblock_side_length or side not in (1, 2):
            raise ValueError("foldblock_side_length must be 1 or 2")
        object.__setattr__(self, "foldblock_side_length", side)

    @property
    def stencil_half_width(self) -> int:
        return self.expansion_order // 2

    @property
    def enclosing_radius(self) -> float:
        """PARSEC ``rmax`` used only to size the underlying Cartesian box."""
        if self.domain_shape == "box":
            return 0.5 * float(np.linalg.norm(np.asarray(self.box_lengths)))
        if self.domain_shape == "ellipsoid":
            # usrinputfile.F90: rmax = maxval(d_shape_param).
            return float(max(self.domain_parameters))
        if self.domain_shape == "cylinder":
            # usrinputfile.F90: rmax = sqrt((length/2)**2 + radius**2).
            radius, length, _ = self.domain_parameters
            return float(np.sqrt((length / 2.0) ** 2 + radius**2))
        return self.radius


@dataclass(frozen=True)
class PeriodicCell:
    """Real-space lattice vectors for a periodic unit cell, in bohr.
    """

    lattice_vectors: np.ndarray
    """Rows are the lattice vectors ``a1``, ``a2``, ``a3``."""

    dimensions: int = 3
    """Periodic directions: 3 (bulk) or 2 (slab: ``a1``, ``a2`` periodic in the xy plane, ``a3 = (0, 0, 2R)`` the
    height of the non-periodic z direction, where the wavefunctions vanish beyond the box)."""

    def __repr__(self) -> str:
        # A bulk cell prints what it printed before slabs had a cell of their own (cache keys hash it).
        return _repr_without_defaults(self, {"dimensions": 3})

    def __post_init__(self) -> None:
        vectors = np.asarray(self.lattice_vectors, dtype=np.float64)
        if vectors.shape != (3, 3):
            raise ValueError("lattice_vectors must have shape (3, 3)")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("lattice_vectors must be finite")
        if abs(float(np.linalg.det(vectors))) <= 0.0:
            raise ValueError("lattice_vectors must span a nondegenerate cell")
        if self.dimensions not in (2, 3):
            raise ValueError("a periodic cell has 2 (slab) or 3 (bulk) periodic dimensions")
        if self.dimensions == 2:
            in_plane = vectors[:2]
            if np.any(np.abs(in_plane[:, 2]) > 1.0e-12) or np.any(np.abs(vectors[2, :2]) > 1.0e-12) or vectors[2, 2] <= 0.0:
                raise ValueError(
                    "a slab cell needs two lattice vectors in the xy plane and a third (0, 0, 2R) along z"
                )
        object.__setattr__(self, "lattice_vectors", vectors)

    @property
    def volume(self) -> float:
        """Unit-cell volume in bohr**3."""
        return float(abs(np.linalg.det(self.lattice_vectors)))


@dataclass(frozen=True)
class PeriodicGridSettings:
    """
    Finite-difference grid settings for a periodic cell.
    Mostly a copy of GridSettings with fewer arguments (may want to fold into GridSettings later)
    """

    spacing: float
    expansion_order: int = 12
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Offset of the first grid point along each lattice axis, in grid steps (Fortran's ``grid%shift``): point
    ``(i, j, k)`` is at ``(i + shift_0) a_0/n_0 + (j + shift_1) a_1/n_1 + (k + shift_2) a_2/n_2``."""
    domain_shape: DomainShape = "PBC_Orthorhombic"
    box_lengths: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # ``Double_Grid_Order``, as for :class:`GridSettings`.
    double_grid_order: int = 1
    # Number of periodic lattice directions: 3 (bulk), 2 (slab: a1, a2
    # periodic, z confined to |z| <= confinement_radius) or 1 (wire: a1 = x
    # periodic, sqrt(y**2 + z**2) <= confinement_radius); F ``pbc%per``.
    periodicity: int = 3
    # ``Boundary_Sphere_Radius`` of a wire or slab (bohr), None for bulk.
    confinement_radius: float | None = None
    # Fortran ``adjust_step`` (inipbc.f90:327-429) on the grids of the
    # Fortran-derived routines: per lattice direction, the fractional
    # translations |M t| of the space group that the step must be
    # commensurate with.  Empty: none (a symmorphic group, Ignore_Symmetry,
    # or no search: molecular dynamics, manual movements).
    fractional_translations: tuple[tuple[float, ...], ...] = ()
    # The local pseudopotential of a 3-D cell: ``"real_space"``, this port's
    # Ewald split (:func:`parsec_python.V_ion.ewald_local_ionic_potential`),
    # or ``"g_space"``, Fortran's reciprocal-space potential of
    # ``v_first.f90``/``pot_local.f90`` (:mod:`parsec_python.V_ion.gspace_local`)
    # with its alpha term ``alpha_term``: ``"analytic"`` (the q -> 0 limit) or
    # ``"finite_difference"`` (Fortran's two-point difference of its q table).
    local_potential: str = "real_space"
    alpha_term: str = "analytic"
    points_rule: str = "round"
    """How the number of points along a lattice vector follows from ``spacing``.  ``"round"``: the nearest integer
    (an orthorhombic cell then needs one common spacing) or the nearest even integer (an oblique cell).  ``"fortran"``:
    the largest even number of points whose step is not below ``spacing``, per direction (Fortran's ``adjust_step``),
    so the steps of a cell may differ.  The input-file parser selects ``"fortran"``."""
    # ``Use_Fortran_Routines`` (parsec_python.Input.routines): ``True`` builds
    # the grid and every periodic term with the Fortran-derived routines of
    # the parity round, ``False`` (the
    # default) with Andrew Grieder's routines.
    fortran_routines: bool = False

    def __repr__(self) -> str:
        # The wire/slab fields, the fractional translations and the local
        # potential print nothing at their defaults, so the text of every
        # earlier calculation stays what it was (cache keys hash it).  The
        # Fortran-derived routines keep that text exactly (the prepared
        # systems of the Fortran parity round are theirs); Andrew Grieder's
        # routines always show their ``points_rule``, so the two never share
        # a cache key.
        hidden = {
            "double_grid_order": 1,
            "periodicity": 3,
            "confinement_radius": None,
            "fractional_translations": (),
            "local_potential": "real_space",
            "alpha_term": "analytic",
            "fortran_routines": self.fortran_routines,
        }
        if self.fortran_routines:
            hidden["points_rule"] = "round"
        return _repr_without_defaults(self, hidden)

    def __post_init__(self) -> None:
        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("grid spacing must be positive")
        periodicity = int(self.periodicity)
        if periodicity != self.periodicity or periodicity not in (1, 2, 3):
            raise ValueError("periodicity must be 1 (wire), 2 (slab) or 3 (bulk)")
        object.__setattr__(self, "periodicity", periodicity)
        if periodicity < 3:
            radius = self.confinement_radius
            if radius is None or not np.isfinite(radius) or radius <= 0:
                raise ValueError(
                    "a wire or slab needs a positive confinement_radius "
                    "(Boundary_Sphere_Radius)"
                )
            object.__setattr__(self, "confinement_radius", float(radius))
        elif self.confinement_radius is not None:
            raise ValueError("confinement_radius applies to a wire or slab only")
        _check_double_grid_order(self)
        translations = tuple(
            tuple(float(value) for value in direction)
            for direction in self.fractional_translations
        )
        if translations and (
            len(translations) != 3
            or any(
                not np.isfinite(value) or value <= 0.0
                for direction in translations
                for value in direction
            )
        ):
            raise ValueError(
                "fractional_translations needs three lists of positive numbers "
                "(one per lattice direction)"
            )
        if not any(translations):
            translations = ()
        object.__setattr__(self, "fractional_translations", translations)
        if self.local_potential not in ("real_space", "g_space"):
            raise ValueError("local_potential must be 'real_space' or 'g_space'")
        if self.alpha_term not in ("analytic", "finite_difference"):
            raise ValueError("alpha_term must be 'analytic' or 'finite_difference'")
        if self.alpha_term != "analytic" and self.local_potential != "g_space":
            raise ValueError(
                "alpha_term belongs to the g_space local potential (the real-space "
                "Ewald split carries the average of the potential itself)"
            )
        if self.local_potential != "real_space" and periodicity != 3:
            raise ValueError(
                "the g_space local potential is that of a 3-D periodic cell; a wire "
                "or slab has its own (Fortran ionpot_wire/ionpot_slab)"
            )
        if self.points_rule not in ("round", "fortran"):
            raise ValueError("points_rule must be 'round' or 'fortran'")
        if self.fortran_routines and self.points_rule != "round":
            raise ValueError(
                "points_rule belongs to the default routines; the Fortran-derived routines "
                "(fortran_routines=True) choose their own points"
            )
        # Both sets of routines read the shift (the Fortran-derived orthorhombic grid since the parity round of
        # PARSEC 1.5; before it started at the origin whatever the shift said).
        if len(self.shift) != 3 or not np.all(np.isfinite(self.shift)):
            raise ValueError("grid shift must have three finite entries")
        object.__setattr__(self, "shift", tuple(float(value) for value in self.shift))
        order = int(self.expansion_order)
        if order != self.expansion_order:
            raise ValueError("PARSEC expansion_order must be an integer")
        if order < 2 or order > 20 or order % 2:
            raise ValueError("PARSEC expansion_order must be an even integer from 2 to 20")
        object.__setattr__(self, "expansion_order", order)

    @property
    def stencil_half_width(self) -> int:
        return self.expansion_order // 2
    @property
    def enclosing_radius(self) -> float:
        """PARSEC ``rmax`` used only to size the underlying Cartesian box."""
        return 0.5 * float(np.linalg.norm(np.asarray(self.box_lengths)))


@dataclass(frozen=True)
class HartreeSettings:
    """Finite-cluster Poisson settings matching ``hartset``/``hpotcg``.

    ``multipole_order`` is PARSEC's ``Solver_Lpole``, the order of the
    multipole expansion of the density that gives the boundary values.  In
    the input it is the smallest order used; in a prepared system it is the
    order in use.

    ``boundary_tolerance`` (Ry) and ``atomic_tail`` control what a prepared
    system does beyond that expansion
    (:mod:`parsec_python.Hartree.boundary`).  The tolerance is the largest
    estimate of the potential the expansion omits for the atoms at the
    boundary: the order is raised until the estimate meets it; ``None``
    switches the estimate off and fixes the order.  ``atomic_tail`` adds the
    omitted terms of the valence point charges at the nuclei: ``"auto"``
    where the estimate asks for it, ``"on"`` always, ``"off"`` never.
    ``boundary_tolerance=None`` with ``atomic_tail="auto"`` is PARSEC's
    boundary, and so is every solver that is called with these settings
    outside a prepared system.

    ``minimum_multipole_order`` is ``None`` in an input.  A prepared system
    whose order was raised keeps the ``Solver_Lpole`` that was asked there,
    so that its input, prepared again, gets the same boundary.

    ``periodic_g_space`` is PARSEC's ``Calc_Hartree_G`` (3-D periodic cells
    only): ``V_H(G) = 8 pi rho(G)/|G|^2`` by FFT
    (:mod:`parsec_python.Hartree.gspace`) instead of the finite-difference
    Poisson solve.  It is not printed by ``repr`` at its default (the
    accelerated driver hashes the settings text).

    ``cg_criterion`` is the stopping test of the conjugate-gradient Poisson
    solve, PARSEC 1.5's ``Hartree_CG_Criterion`` (``ipar(3)`` of SPARSKIT's
    ``cg`` in ``potential/hartree.F90:2096-2108``): 1 (the default) stops at
    ``||r_k|| <= relative_tolerance*||r_0|| + absolute_tolerance``, 2 at
    ``||r_k|| <= relative_tolerance*||b|| + absolute_tolerance`` with ``b``
    the right-hand side (:func:`parsec_python.Hartree.poisson.cg_stopping_threshold`).  It is not printed by
    ``repr`` at its default.
    """

    multipole_order: int = 9
    relative_tolerance: float = 1.0e-7
    absolute_tolerance: float = 1.0e-13
    max_iterations: int = 1600
    boundary_method: HartreeBoundaryMethod = "auto"
    direct_chunk_size: int = 16
    boundary_tolerance: float | None = 1.0e-3
    atomic_tail: HartreeAtomicTail = "auto"
    minimum_multipole_order: int | None = None
    periodic_g_space: bool = False
    cg_criterion: int = 1

    def __repr__(self) -> str:
        return _repr_without_defaults(self, {"periodic_g_space": False, "cg_criterion": 1})

    def __post_init__(self) -> None:
        order = int(self.multipole_order)
        max_iterations = int(self.max_iterations)
        direct_chunk_size = int(self.direct_chunk_size)
        if (
            order != self.multipole_order
            or not 0 <= order <= MAXIMUM_MULTIPOLE_ORDER
        ):
            raise ValueError(
                f"multipole_order must be between 0 and {MAXIMUM_MULTIPOLE_ORDER}"
            )
        if self.boundary_tolerance is not None and not (
            np.isfinite(self.boundary_tolerance) and self.boundary_tolerance > 0
        ):
            raise ValueError(
                "Hartree boundary_tolerance must be positive, or None for off"
            )
        if self.atomic_tail not in {"auto", "on", "off"}:
            raise ValueError("Hartree atomic_tail must be auto, on, or off")
        minimum = self.minimum_multipole_order
        if minimum is not None and not (
            int(minimum) == minimum and 0 <= minimum <= order
        ):
            raise ValueError(
                "minimum_multipole_order must be between 0 and multipole_order"
            )
        if (
            not np.isfinite(self.relative_tolerance)
            or not np.isfinite(self.absolute_tolerance)
            or self.relative_tolerance <= 0
            or self.absolute_tolerance < 0
        ):
            raise ValueError("invalid Hartree solver tolerance")
        if max_iterations != self.max_iterations or max_iterations < 1:
            raise ValueError("Hartree max_iterations must be positive")
        if self.cg_criterion not in (1, 2):
            raise ValueError("Hartree cg_criterion must be 1 (initial residual) or 2 (right-hand side)")
        if self.boundary_method not in {"auto", "multipole", "direct"}:
            raise ValueError("Hartree boundary_method must be auto, multipole, or direct")
        if (
            direct_chunk_size != self.direct_chunk_size
            or direct_chunk_size < 1
        ):
            raise ValueError("direct_chunk_size must be a positive integer")
        object.__setattr__(self, "multipole_order", order)
        if minimum is not None:
            object.__setattr__(self, "minimum_multipole_order", int(minimum))
        object.__setattr__(self, "max_iterations", max_iterations)
        object.__setattr__(self, "direct_chunk_size", direct_chunk_size)


@dataclass(frozen=True)
class EigensolverSettings:
    """PARSEC eigensolver controls retained by the pure-Python port.

    ``first_filter_degree`` is PARSEC's ``Chebdav_Degree`` for CHEBFF and
    CHEBDAV.  ``first_filter_cycles`` is ``FF_MaxIter`` (default 3, PARSEC
    1.5's, since 2026-11-04; 2 before) and affects CHEBFF
    only; CHEBDAV instead uses ``tolerance`` for residual locking.
    ``filter_degree`` and ``filter_degree_delta`` control the later subspace
    filter.  The input translator resolves PARSEC's ``Matvec_Blocksize``
    default from ``States_Num`` and stores the resulting positive integer in
    ``matvec_block_size``.

    ``method`` may also be ``arpack`` (``arpk.f90z``), ``diagla``
    (``diagla.f90z``) or ``none`` (PARSEC's TEST mode: setup only).
    ``chebyshev_filtering`` is ``Chebyshev_Filtering``: ``None`` keeps the
    rule of ``usrinputfile.F90`` (always on for CHEBFF and CHEBDAV, off
    otherwise); for ARPACK and DIAGLA ``True`` runs the selected solver for
    the first solve only and the later-SCF SUBSPACE filter afterwards, as
    after CHEBFF, ``False`` runs the solver in every SCF iteration.
    ``dynamic_tolerance`` is ``Dynamic_Diag_Tol``: the first SCF iteration
    uses a tolerance of 0.1 and every later one
    ``max(min(tol, min_s SRE_s / 10), tolerance)`` (parsec.F90:1304-1311,
    the rule Fortran intends; see :mod:`parsec_python.Eigensolvers.eigval`).
    ``maximum_matvec`` is ``Maximum_matvec``: ARPACK and DIAGLA fail when a
    solve uses ``max(maximum_matvec, n_grid * n_states)`` matrix-vector
    products or more (eigval.F90 ierr 402).  ``fixed_state_count`` is
    ``Fix_Number_Eigenvalues`` of a ``Restart_Run``: PARSEC skips
    ``initeigval`` and keeps the number of eigenpairs per representation that
    the restart file gave (``nn = nec`` of ``restart_run.F90``, the stored
    ``States_Num`` levels; ``eigval.F90:679-685`` resets ``nn = nec = mm``
    after every solve), so the eigensolver works with ``States_Num`` vectors
    and no ``subspace_buffer`` (see :attr:`working_buffer`).  The new fields
    print nothing at their defaults, so the text of the settings of every
    earlier calculation is unchanged (it enters cache keys of the
    accelerated driver).

    Options of the PARSEC 1.5 round (all off by default, printed only when
    set): ``vector_seed`` is ``Vector_Rng_Seed``: the start vectors of the
    first solve (CHEBFF/CHEBDAV) come from the LAPACK seed it selects
    (:func:`parsec_python.Eigensolvers.lapack_random.lapack_seed_from_integer`)
    and ``random_seed`` is set to it by the input reader, so the Lanczos
    starts follow too; ``None`` keeps PARSEC's canonical seed.
    ``reorthogonalization_threshold`` is the first ratio of the selective
    Gram--Schmidt (0.1, the earlier PARSEC; 1.5 uses 0.68).
    ``first_filter_residual_test`` is PARSEC 1.5's residual-driven CHEBFF
    (``FF_Residual_Test``): ``"off"`` (fixed cycles, the default), ``"stop"``
    (1.5: the first-solve cycles stop once the largest residual of the
    ``States_Num`` states is below the eigensolver tolerance of the
    iteration, and the run stops, 1.5 ierr 401, when ``first_filter_cycles``
    cycles do not reach it) or ``"continue"`` (the same early exit; the SCF
    goes on after ``first_filter_cycles`` cycles otherwise).
    ``exact_state_count`` is PARSEC 1.5's working space of exactly
    ``States_Num`` vectors (``Exact_State_Count: .true.``; 1.5 has no buffer at
    all, ``eigen_solver_module.F90:53``).  The PARSEC 1.5 SUBSPACE rules
    (readable eigensolvers only): ``subspace_lower_bound`` ``"monotone"``
    (the earlier PARSEC: the largest Ritz value, never lowered), ``"smart"``
    or ``"adaptive"`` (``Smart_Subspace`` true or false,
    ``subspace.f90z:237-327`` of 1.5); ``scaled_subspace_filter``
    (``Chebyshev_Scaled``: the filter normalized at the lowest Ritz value);
    ``lanczos_bounds`` ``"parsec"`` (5 SUBSPACE and 8 CHEBFF steps, the
    earlier bound), ``"lanczos"`` or ``"lanczos_full"`` (``Bounds_Method`` of
    1.5: ``max(degree/2, lanczos_bound_steps)`` steps and 1.5's bound,
    ``lancz_bound.f90z`` of 1.5; ``lanczos_full`` with full
    reorthogonalization), ``lanczos_bound_steps`` being ``Bounds_Iter``.
    ``filter_restart`` is ``(Filter_Restart_After, Restart_Cheb_Remained_Times,
    Remained_Times_Atomic_Chrg_Den)`` of PARSEC 1.5 for relaxation and
    molecular dynamics (:mod:`parsec_python.SCF.filter_restart`), ``None``
    without it.
    """

    method: EigensolverName = "chebff"
    tolerance: float = 1.0e-4
    first_filter_degree: int = 20
    first_filter_cycles: int = 3
    matvec_block_size: int = 6
    subspace_buffer: int = 6
    filter_degree: int = 15
    filter_degree_delta: int = 3
    lanczos_steps: int = 5
    random_seed: int = 7
    chebyshev_filtering: bool | None = None
    dynamic_tolerance: bool = False
    maximum_matvec: int = DEFAULT_MAXIMUM_MATVEC
    fixed_state_count: bool = False
    vector_seed: int | None = None
    reorthogonalization_threshold: float = 0.1
    first_filter_residual_test: str = "off"
    exact_state_count: bool = False
    subspace_lower_bound: str = "monotone"
    scaled_subspace_filter: bool = False
    lanczos_bounds: str = "parsec"
    lanczos_bound_steps: int = 10
    filter_restart: tuple[int, int, int] | None = None

    def __repr__(self) -> str:
        # Hidden at their defaults: the text of these settings is hashed by
        # the cache keys of the accelerated driver.
        hidden = {
            "chebyshev_filtering": self.chebyshev_filtering is None,
            "dynamic_tolerance": not self.dynamic_tolerance,
            "maximum_matvec": self.maximum_matvec == DEFAULT_MAXIMUM_MATVEC,
            "fixed_state_count": not self.fixed_state_count,
            "vector_seed": self.vector_seed is None,
            "reorthogonalization_threshold": self.reorthogonalization_threshold == 0.1,
            "first_filter_residual_test": self.first_filter_residual_test == "off",
            "exact_state_count": not self.exact_state_count,
            "subspace_lower_bound": self.subspace_lower_bound == "monotone",
            "scaled_subspace_filter": not self.scaled_subspace_filter,
            "lanczos_bounds": self.lanczos_bounds == "parsec",
            "lanczos_bound_steps": self.lanczos_bound_steps == 10,
            "filter_restart": self.filter_restart is None,
        }
        shown = [
            f"{item.name}={getattr(self, item.name)!r}"
            for item in fields(self)
            if not hidden.get(item.name, False)
        ]
        return f"{type(self).__qualname__}({', '.join(shown)})"

    @property
    def filters_later_iterations(self) -> bool:
        """Whether later SCF iterations run the Chebyshev SUBSPACE filter.

        ``solver%do_subsp`` of ``usrinputfile.F90``: forced on for CHEBFF
        and CHEBDAV, ``Chebyshev_Filtering`` (default off) otherwise.
        """

        if self.method in ALWAYS_FILTERING_EIGENSOLVERS:
            return True
        return bool(self.chebyshev_filtering)

    @property
    def working_buffer(self) -> int:
        """Working vectors beyond ``States_Num`` (the eigensolver's buffer).

        ``subspace_buffer``, or none with ``fixed_state_count``
        (``Fix_Number_Eigenvalues``: PARSEC keeps the stored number of
        eigenpairs, ``States_Num``, in every solve).
        """

        return 0 if self.fixed_state_count or self.exact_state_count else self.subspace_buffer

    @property
    def basis_seed(self) -> tuple[int, int, int, int]:
        """The LAPACK seed of the first-solve start vectors (``Vector_Rng_Seed``)."""

        from .Eigensolvers.lapack_random import (
            PARSEC_RANDOM_ARRAY_SEED,
            lapack_seed_from_integer,
        )

        if self.vector_seed is None:
            return PARSEC_RANDOM_ARRAY_SEED
        return lapack_seed_from_integer(self.vector_seed)

    def __post_init__(self) -> None:
        if self.method not in {"chebff", "chebdav", "arpack", "diagla", "none"}:
            raise ValueError(
                "eigensolver method must be 'chebff', 'chebdav', 'arpack', "
                "'diagla', or 'none'"
            )
        if self.chebyshev_filtering is not None:
            object.__setattr__(
                self, "chebyshev_filtering", bool(self.chebyshev_filtering)
            )
            if (
                self.method in ALWAYS_FILTERING_EIGENSOLVERS
                and not self.chebyshev_filtering
            ):
                raise ValueError(
                    f"PARSEC always runs the Chebyshev subspace filter after "
                    f"{self.method}; chebyshev_filtering cannot be False"
                )
        object.__setattr__(self, "dynamic_tolerance", bool(self.dynamic_tolerance))
        object.__setattr__(self, "fixed_state_count", bool(self.fixed_state_count))
        object.__setattr__(self, "exact_state_count", bool(self.exact_state_count))
        object.__setattr__(
            self, "scaled_subspace_filter", bool(self.scaled_subspace_filter)
        )
        if self.first_filter_residual_test not in ("off", "stop", "continue"):
            raise ValueError("first_filter_residual_test must be off, stop or continue")
        if self.subspace_lower_bound not in ("monotone", "smart", "adaptive"):
            raise ValueError("subspace_lower_bound must be monotone, smart or adaptive")
        if self.lanczos_bounds not in ("parsec", "lanczos", "lanczos_full"):
            raise ValueError("lanczos_bounds must be parsec, lanczos or lanczos_full")
        steps = int(self.lanczos_bound_steps)
        if steps != self.lanczos_bound_steps or steps < 1:
            raise ValueError("lanczos_bound_steps must be a positive integer")
        object.__setattr__(self, "lanczos_bound_steps", steps)
        if self.filter_restart is not None:
            restart = tuple(int(item) for item in self.filter_restart)
            if len(restart) != 3 or restart[0] < 1:
                raise ValueError(
                    "filter_restart is (Filter_Restart_After >= 1, restarts, density resets)"
                )
            object.__setattr__(self, "filter_restart", restart)
        if self.vector_seed is not None:
            if int(self.vector_seed) != self.vector_seed:
                raise ValueError("vector_seed must be an integer")
            object.__setattr__(self, "vector_seed", int(self.vector_seed))
        threshold = float(self.reorthogonalization_threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError("reorthogonalization_threshold must be in (0, 1)")
        object.__setattr__(self, "reorthogonalization_threshold", threshold)
        if self.first_filter_residual_test != "off" and self.method != "chebff":
            raise ValueError("first_filter_residual_test is a CHEBFF option")
        maximum_matvec = int(self.maximum_matvec)
        if maximum_matvec != self.maximum_matvec or maximum_matvec < 1:
            raise ValueError("maximum_matvec must be a positive integer")
        object.__setattr__(self, "maximum_matvec", maximum_matvec)
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("eigensolver tolerance must be positive")
        integer_fields = (
            "first_filter_degree",
            "first_filter_cycles",
            "matvec_block_size",
            "subspace_buffer",
            "filter_degree",
            "filter_degree_delta",
            "lanczos_steps",
            "random_seed",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if int(value) != value:
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        minimum_first_degree = 15 if self.method == "chebdav" else 10
        # Chebdav_Degree belongs to the CHEBFF and CHEBDAV first solves;
        # usrinputfile.F90 does not read it for the other eigensolvers.
        if (
            self.method in ALWAYS_FILTERING_EIGENSOLVERS
            and self.first_filter_degree < minimum_first_degree
        ):
            raise ValueError(
                f"first_filter_degree must be at least {minimum_first_degree} "
                f"for {self.method}"
            )
        # 1..9 in the earlier PARSEC, up to 50 in PARSEC 1.5
        # (usrinputfile.F90:2303 of 1.5).
        if not 1 <= self.first_filter_cycles <= 50:
            raise ValueError("first_filter_cycles must be between 1 and 50")
        if self.matvec_block_size < 1:
            raise ValueError("matvec_block_size must be positive")
        if self.subspace_buffer < 0:
            raise ValueError("subspace_buffer cannot be negative")
        if (
            self.filters_later_iterations
            and self.subspace_buffer < 6
            and not self.exact_state_count
        ):
            raise ValueError(
                "PARSEC filtering requires a subspace buffer of at least 6"
            )
        for name in ("filter_degree", "lanczos_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.filter_degree_delta < self.filter_degree:
            raise ValueError("filter_degree_delta must be nonnegative and smaller than filter_degree")
        if self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative")


# PARSEC's Mixing_Method values (const.f90: ANDERSON, BROYDEN, MSECANT1-3).
MIXING_METHODS = ("anderson", "broyden", "msecant1", "msecant2", "msecant3")


@dataclass(frozen=True)
class MixingSettings:
    """Potential mixing settings used by the SCF driver.

    ``method`` is PARSEC's ``Mixing_Method``: ``anderson`` (the default,
    :class:`parsec_python.Mixer.AndersonMixer`), ``broyden`` (the generalized
    Broyden mixer of ``genbro.f90``) or ``msecant1``/``msecant2``/``msecant3``
    (the multisecant mixers of ``msecant1-3.f90``, Type-I, Type-II and the
    hybrid of the two); :func:`parsec_python.Mixer.make_mixer` builds it.
    ``parameter`` (``Mixing_Param``), ``memory`` (``Memory_Param``) and
    ``restart`` (``Restart_Mixing``) are read by every method as PARSEC
    reads them; the multisecant mixers take ``memory`` as the first size of
    their history only and never read ``restart``.  The last five fields are
    the multisecant keywords of ``usrinputfile.F90:1584-1600`` with their
    PARSEC defaults: ``group_size`` (``Mixing_Group_Size``, 0 for one group
    of every secant pair), ``en_like`` (``Mixing_EN_Like``), ``restart_factor``
    (``Mixing_Restart_Factor``), ``expand_factor``
    (``Mixing_Memory_Expand_Factor``) and ``preferred_type``
    (``Mixing_Preferred_Type``, 1 for Type-I, any other value Type-II).

    The Anderson mixer keeps its validation: ``parameter`` in (0, 1],
    positive ``memory`` and ``restart``.  PARSEC checks none of these numbers;
    for the other methods only what their algebra needs is required: a
    finite ``parameter``, a nonnegative ``memory`` (0 is linear mixing for
    the Broyden mixer) and, for the Broyden mixer, a positive ``restart``.
    ``regularization`` and the safeguard are options of the Anderson mixer
    of this port and are refused for the others.
    """

    parameter: float = 0.30
    memory: int = 4
    restart: int = 20
    regularization: float = 0.0
    safeguard: bool = False
    step_limit: float = 2.0
    growth_trigger: float = 2.0
    backoff: float = 0.5
    method: str = "anderson"
    group_size: int = 1
    en_like: int = 0
    restart_factor: float = 0.3
    expand_factor: float = 2.0
    preferred_type: int = 1

    def __repr__(self) -> str:
        # The text of these settings is hashed by the cache keys of the
        # accelerated driver and stored in test records.  The method and the
        # multisecant fields at their defaults print nothing, so that the
        # text of every Anderson calculation stays what it was before the
        # fields existed.
        defaults = {
            "method": "anderson",
            "group_size": 1,
            "en_like": 0,
            "restart_factor": 0.3,
            "expand_factor": 2.0,
            "preferred_type": 1,
        }
        shown = [
            f"{item.name}={getattr(self, item.name)!r}"
            for item in fields(self)
            if not (
                item.name in defaults
                and getattr(self, item.name) == defaults[item.name]
            )
        ]
        return f"{type(self).__qualname__}({', '.join(shown)})"

    def __post_init__(self) -> None:
        method = str(self.method).strip().lower()
        if method not in MIXING_METHODS:
            raise ValueError(
                f"unknown mixing method {self.method!r}; PARSEC's mixers are "
                + ", ".join(MIXING_METHODS)
            )
        object.__setattr__(self, "method", method)
        if method == "anderson":
            if not np.isfinite(self.parameter) or not 0 < self.parameter <= 1:
                raise ValueError("mixing parameter must be in (0, 1]")
            memory = int(self.memory)
            restart = int(self.restart)
            if (
                memory != self.memory
                or restart != self.restart
                or memory < 1
                or restart < 1
            ):
                raise ValueError("mixing memory and restart must be positive")
        else:
            if not np.isfinite(self.parameter):
                raise ValueError("mixing parameter must be finite")
            memory = int(self.memory)
            restart = int(self.restart)
            if memory != self.memory or restart != self.restart:
                raise ValueError("mixing memory and restart must be integers")
            if memory < 0:
                raise ValueError("mixing memory cannot be negative")
            if method == "broyden" and restart < 1:
                raise ValueError("Broyden mixing restart must be positive")
            if self.regularization or self.safeguard:
                raise ValueError(
                    "mixing regularization and the safeguard are options of "
                    "the Anderson mixer only"
                )
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("mixing regularization cannot be negative")
        if not np.isfinite(self.step_limit) or self.step_limit <= 0:
            raise ValueError("mixing safeguard step limit must be positive")
        if not np.isfinite(self.growth_trigger) or self.growth_trigger <= 1:
            raise ValueError("mixing safeguard growth trigger must exceed one")
        if not np.isfinite(self.backoff) or not 0 < self.backoff <= 1:
            raise ValueError("mixing safeguard backoff must be in (0, 1]")
        object.__setattr__(self, "memory", memory)
        object.__setattr__(self, "restart", restart)
        integers = {}
        for name in ("group_size", "en_like", "preferred_type"):
            value = getattr(self, name)
            integer = int(value)
            if integer != value:
                raise ValueError(f"mixing {name} must be an integer")
            integers[name] = integer
        if integers["group_size"] < 0:
            raise ValueError("mixing group size cannot be negative")
        for name in ("restart_factor", "expand_factor"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"mixing {name} must be finite")
        for name, value in integers.items():
            object.__setattr__(self, name, value)


#: ``Exx_Max_Iter``: outer (Fock) iterations of a hybrid SCF.  100 is the cap of
#: Quantum ESPRESSO's outer loop (``electron_maxstep``): a state with a soft mode
#: converges linearly and slowly (a ring of 12 hydrogen atoms, bond 1.40 bohr,
#: PBE0: dexx shrinks by 0.90 per outer iteration here and in QE 7.5, which needs 66).
DEFAULT_EXX_MAX_ITERATIONS = 100
#: ``Exx_Tolerance`` (Ry): the estimated exchange error of the Fock loop
#: (Quantum ESPRESSO's ``dexx``) below which it stops.
DEFAULT_EXX_TOLERANCE = 1.0e-6
#: ``Exx_Batch_Size``: pair densities per batched Poisson solve.
DEFAULT_EXX_BATCH_SIZE = 32


@dataclass(frozen=True)
class ExactExchangeSettings:
    """Controls of a self-consistent hybrid functional (``SCFSettings.exact_exchange``).

    ``fraction`` is ``Exact_Exchange_Fraction`` and ``screening`` is
    ``Screening_Parameter`` (bohr^-1); ``None`` takes the value of the
    functional (:mod:`parsec_python.SCF.hybrid`).  ``max_iterations``
    (``Exx_Max_Iter``) bounds the outer Fock iterations, ``tolerance``
    (``Exx_Tolerance``, Ry) is the estimated exchange error at which they
    stop, and ``batch_size`` (``Exx_Batch_Size``) is the number of pair
    densities each Poisson call solves.  ``pair_solver`` (``Exx_Pair_Solver``)
    is ``"kronecker"`` (the direct solver on the bounding box, every kernel)
    or ``"cg"`` (the conjugate gradients of the Hartree solver, 1/r only).
    The text of the settings names only what differs from the defaults.
    """

    fraction: float | None = None
    screening: float | None = None
    max_iterations: int = DEFAULT_EXX_MAX_ITERATIONS
    tolerance: float = DEFAULT_EXX_TOLERANCE
    batch_size: int = DEFAULT_EXX_BATCH_SIZE
    pair_solver: str = "kronecker"

    def __post_init__(self) -> None:
        if self.pair_solver not in ("kronecker", "cg"):
            raise ValueError("the pair solver must be 'kronecker' or 'cg'")
        if self.fraction is not None:
            fraction = float(self.fraction)
            if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
                raise ValueError("the exact-exchange fraction must lie in (0, 1]")
            object.__setattr__(self, "fraction", fraction)
        if self.screening is not None:
            screening = float(self.screening)
            if not np.isfinite(screening) or screening <= 0.0:
                raise ValueError("the screening parameter must be positive (bohr^-1)")
            object.__setattr__(self, "screening", screening)
        iterations = int(self.max_iterations)
        if iterations != self.max_iterations or iterations < 1:
            raise ValueError("the Fock-loop iteration limit must be a positive integer")
        object.__setattr__(self, "max_iterations", iterations)
        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("the Fock-loop tolerance must be positive (Ry)")
        object.__setattr__(self, "tolerance", tolerance)
        batch = int(self.batch_size)
        if batch != self.batch_size or batch < 1:
            raise ValueError("the pair-density batch size must be a positive integer")
        object.__setattr__(self, "batch_size", batch)

    def __repr__(self) -> str:
        defaults = {
            "fraction": None,
            "screening": None,
            "max_iterations": DEFAULT_EXX_MAX_ITERATIONS,
            "tolerance": DEFAULT_EXX_TOLERANCE,
            "batch_size": DEFAULT_EXX_BATCH_SIZE,
            "pair_solver": "kronecker",
        }
        shown = [
            f"{item.name}={getattr(self, item.name)!r}"
            for item in fields(self)
            if getattr(self, item.name) != defaults[item.name]
        ]
        return f"{type(self).__qualname__}({', '.join(shown)})"


@dataclass(frozen=True)
class SCFSettings:
    """Isolated single-point SCF controls.

    ``xc_functional`` is PARSEC's ``Correlation_Type`` (the names and the
    Fortran labels are listed in :mod:`parsec_python.V_xc.selection`);
    ``xalpha``, ``wigner`` and ``hedin_lundqvist`` have no spin-polarized
    form.  ``ionization_energy_difference`` is ``Ion_Energy_Diff`` in
    Rydberg, the cap of the ``casida_salahub`` potential correction, and is
    read by that functional only.

    ``fixed_occupations`` holds one occupation per state of
    ``number_of_states``, in ascending order of energy and with one standing
    for a doubly occupied orbital.  It is what PARSEC reads from ``occup.in``
    for a negative ``Fermi_Temp``; when given, no Fermi level is searched and
    the temperature is not used.

    ``spin_polarized`` is PARSEC's ``Spin_Polarization``: two spin channels,
    each with its own potential, eigenproblem and ``number_of_states``
    states, and one Fermi level for both.  A state of a channel holds one
    electron, so an occupation of one stands for one electron there.
    ``fixed_occupations`` are then those of the up channel (the first
    section of ``occup.in``) and ``fixed_occupations_down`` those of the
    down channel (its second section).

    For an isolated input without spin-orbit coupling, and for a periodic cell at the Gamma point whose
    forces, relaxation or dynamics are asked for, ``spin_polarized`` is run by
    :func:`~parsec_python.SCF.spin_polarized.run_spin_scf`; the periodic energy-only runs, spin-orbit coupling and
    non-collinear magnetism use
    :func:`~parsec_python.SCF.spin_polarized.run_scf_spin_polarized` and the
    k-point drivers.

    ``self_consistent_spin_orbit`` mirrors PARSEC's ``SO_from_scratch``
    (and, as a deliberate simplification of Fortran's more convoluted
    ``Non_Collinear_magnetism``-gated logic, ``SCF_SO`` treated the same
    way): implemented by
    :func:`~parsec_python.SCF.self_consistent_soc.run_self_consistent_soc`
    for isolated inputs with at least one ``SO_PSP=true`` species and
    ``spin_polarized=False`` (both spinor components share one
    spin-unpolarized potential), and by the periodic k-point drivers.

    ``noncollinear`` mirrors PARSEC's ``Non_Collinear_magnetism``: the
    magnetization is a vector field ``m(r)`` whose direction varies in space
    (see :mod:`parsec_python.SCF.noncollinear`).  Spin-orbit coupling is
    added on top when ``self_consistent_spin_orbit`` is also set (as in
    Fortran, a species with ``SO_PSP`` together with ``Non_Collinear_magnetism``
    switches the self-consistent spin-orbit term on).

    ``symmetrize_density`` is PARSEC's ``Symmetrize_Charge_Density``: every
    new density of an isolated SCF loop is replaced by its average over the
    point group of the atoms (``symm_scalar`` of ``newrho.F90``), and every
    new density of a periodic loop by its average over the space group of
    ``SinglePointInput.space_group`` (:mod:`parsec_python.Grid.space_group`).

    ``van_der_waals`` is PARSEC's ``Van_Der_Waals``: the Tkatchenko-Scheffler
    dispersion correction evaluated once on the converged density
    (:mod:`parsec_python.Energy.dispersion`).  ``hirshfeld_cells`` are
    ``Hf_One``, ``Hf_Two`` and ``Hf_Three`` of a periodic cell: the free-atom
    densities of the Hirshfeld partition are summed over
    ``(2*Hf_One+1)*(2*Hf_Two+1)*(2*Hf_Three+1)`` cells.
    """

    max_iterations: int = 50
    convergence_criterion: float = 2.0e-4
    fermi_temperature_kelvin: float = 80.0
    number_of_states: int | None = None
    net_charge: float = 0.0
    use_plain_residual: bool = False
    normalize_initial_density: bool = True
    xc_functional: XCFunctional = "ca"
    fixed_occupations: tuple[float, ...] | None = None
    spin_polarized: bool = False
    fixed_occupations_down: tuple[float, ...] | None = None
    self_consistent_spin_orbit: bool = False
    noncollinear: bool = False
    symmetrize_density: bool = False
    van_der_waals: bool = False
    hirshfeld_cells: tuple[int, int, int] = (0, 0, 0)
    # ``Convergence_Criterion_Approach`` (Ry): ``None`` is PARSEC's default
    # ``100 * Convergence_Criterion`` (usrinputfile.F90), see
    # :attr:`approach_criterion`.
    convergence_criterion_approach: float | None = None
    ionization_energy_difference: float = 0.0
    # Fortran's ``potshift`` (rho_hart.F90 l.243-257, a 3-D periodic cell):
    # the exchange-correlation potential of each spin channel is shifted to
    # a zero average every time it is evaluated (the Hartree potential has a
    # zero average anyway).  Applied by the periodic prepared system.
    zero_average_xc_potential: bool = False
    # The later-SCF Chebyshev degree is lowered from the iteration after this
    # one on: 4, PARSEC 1.5's ``iter > 4`` (parsec.F90:1942-1952 of 1.5), the
    # default since 2026-11-04 (5, the earlier PARSEC's
    # ``iter > 5`` of parsec.F90:1453, gives the runs before it bit for bit);
    # see :attr:`approach_criterion`.
    degree_decrease_after: int = 4
    # The Fock-loop controls of a hybrid ``xc_functional``
    # (:class:`ExactExchangeSettings`); ``None`` for every other functional.
    exact_exchange: ExactExchangeSettings | None = None

    @property
    def is_hybrid(self) -> bool:
        """Whether ``xc_functional`` has a nonlocal exact-exchange term."""

        return self.xc_functional in HYBRID_XC_FUNCTIONALS

    def __repr__(self) -> str:
        # The text of these settings is hashed by the cache keys of the
        # accelerated driver and stored in test records.  Occupations that
        # are not set, and spin or spin-orbit switches that are off, print
        # nothing, so that the text of every calculation without them stays
        # what it was before the fields existed.
        hidden = {
            "fixed_occupations": self.fixed_occupations is None,
            "spin_polarized": not self.spin_polarized,
            "fixed_occupations_down": self.fixed_occupations_down is None,
            "self_consistent_spin_orbit": not self.self_consistent_spin_orbit,
            "noncollinear": not self.noncollinear,
            "symmetrize_density": not self.symmetrize_density,
            "van_der_waals": not self.van_der_waals,
            "hirshfeld_cells": tuple(self.hirshfeld_cells) == (0, 0, 0),
            "convergence_criterion_approach": (
                self.convergence_criterion_approach is None
            ),
            "ionization_energy_difference": self.ionization_energy_difference == 0.0,
            "zero_average_xc_potential": not self.zero_average_xc_potential,
            "degree_decrease_after": self.degree_decrease_after == 4,
            "exact_exchange": self.exact_exchange is None,
        }
        shown = [
            f"{item.name}={getattr(self, item.name)!r}"
            for item in fields(self)
            if not hidden.get(item.name, False)
        ]
        return f"{type(self).__qualname__}({', '.join(shown)})"

    @property
    def approach_criterion(self) -> float:
        """The SRE below which later SCF iterations lower the filter degree.

        ``vconv_approach`` of parsec.F90:1452-1454: once the iteration is
        beyond the fourth (:attr:`degree_decrease_after`) and the weighted SRE
        (the larger of the spin channels) falls below it, the later-SCF
        Chebyshev degree drops by one per iteration, down to 10.  The default
        ``100 * vconv`` follows the criterion of this loop
        (``Relaxation_SCF_Upscale`` replaces it); PARSEC 1.5's default is
        ``2 * vconv`` (usrinputfile.F90:2596-2601 of 1.5), which an explicit
        ``Convergence_Criterion_Approach`` gives.
        """

        if self.convergence_criterion_approach is None:
            return 100.0 * self.convergence_criterion
        return self.convergence_criterion_approach

    def __post_init__(self) -> None:
        max_iterations = int(self.max_iterations)
        if max_iterations != self.max_iterations or max_iterations < 1:
            raise ValueError("SCF max_iterations must be positive")
        if (
            not np.isfinite(self.convergence_criterion)
            or self.convergence_criterion <= 0
        ):
            raise ValueError("SCF convergence criterion must be positive")
        if not np.isfinite(self.fermi_temperature_kelvin) or (
            self.fermi_temperature_kelvin < 0 and self.fixed_occupations is None
        ):
            raise ValueError(
                "a negative Fermi temperature selects fixed occupations "
                "(PARSEC's occup.in): give fixed_occupations"
            )
        if not np.isfinite(self.net_charge):
            raise ValueError("net_charge must be finite")
        decrease_after = int(self.degree_decrease_after)
        if decrease_after != self.degree_decrease_after or decrease_after < 0:
            raise ValueError("degree_decrease_after must be a nonnegative integer")
        object.__setattr__(self, "degree_decrease_after", decrease_after)
        if self.convergence_criterion_approach is not None:
            approach = float(self.convergence_criterion_approach)
            if not np.isfinite(approach):
                raise ValueError("convergence_criterion_approach must be finite")
            object.__setattr__(self, "convergence_criterion_approach", approach)
        if self.xc_functional not in XC_FUNCTIONAL_NAMES:
            raise ValueError(
                "xc_functional must be one of "
                + ", ".join(repr(name) for name in sorted(XC_FUNCTIONAL_NAMES))
            )
        if self.xc_functional in UNPOLARIZED_ONLY_XC_FUNCTIONALS and (
            self.spin_polarized or self.noncollinear
        ):
            raise ValueError(
                f"xc_functional={self.xc_functional!r} has no spin-polarized "
                "form; it cannot be used with spin polarization or "
                "non-collinear magnetism"
            )
        if self.xc_functional in HYBRID_XC_FUNCTIONALS:
            if self.exact_exchange is None:
                object.__setattr__(self, "exact_exchange", ExactExchangeSettings())
            elif not isinstance(self.exact_exchange, ExactExchangeSettings):
                raise TypeError("exact_exchange must be an ExactExchangeSettings")
        elif self.exact_exchange is not None:
            raise ValueError(
                "exact_exchange settings belong to a hybrid xc_functional, not "
                f"to {self.xc_functional!r}"
            )
        energy_difference = float(self.ionization_energy_difference)
        if not np.isfinite(energy_difference):
            raise ValueError("ionization_energy_difference must be finite")
        if energy_difference != 0.0 and self.xc_functional != "casida_salahub":
            raise ValueError(
                "ionization_energy_difference is read by the casida_salahub "
                "functional only"
            )
        object.__setattr__(self, "ionization_energy_difference", energy_difference)
        if self.number_of_states is not None:
            states = int(self.number_of_states)
            if states != self.number_of_states or states < 1:
                raise ValueError("number_of_states must be a positive integer")
            object.__setattr__(self, "number_of_states", states)
        if self.fixed_occupations is not None:
            fixed = tuple(float(value) for value in self.fixed_occupations)
            if (
                self.number_of_states is None
                or len(fixed) != self.number_of_states
                or not np.all(np.isfinite(fixed))
            ):
                raise ValueError(
                    "fixed_occupations needs one finite value per state of "
                    "number_of_states"
                )
            object.__setattr__(self, "fixed_occupations", fixed)
        object.__setattr__(self, "spin_polarized", bool(self.spin_polarized))
        object.__setattr__(self, "symmetrize_density", bool(self.symmetrize_density))
        if self.fixed_occupations_down is not None:
            if not self.spin_polarized or self.fixed_occupations is None:
                raise ValueError(
                    "fixed_occupations_down belongs to a spin-polarized "
                    "calculation with fixed_occupations for the up channel"
                )
            down = tuple(float(value) for value in self.fixed_occupations_down)
            if len(down) != self.number_of_states or not np.all(np.isfinite(down)):
                raise ValueError(
                    "fixed_occupations_down needs one finite value per state "
                    "of number_of_states"
                )
            object.__setattr__(self, "fixed_occupations_down", down)
        elif self.spin_polarized and self.fixed_occupations is not None:
            raise ValueError(
                "a spin-polarized calculation with fixed occupations needs "
                "them for both channels: give fixed_occupations_down"
            )
        object.__setattr__(self, "van_der_waals", bool(self.van_der_waals))
        cells = tuple(int(value) for value in self.hirshfeld_cells)
        if len(cells) != 3 or any(
            cell != value or cell < 0
            for cell, value in zip(cells, self.hirshfeld_cells)
        ):
            raise ValueError("hirshfeld_cells must be three non-negative integers")
        object.__setattr__(self, "hirshfeld_cells", cells)
        object.__setattr__(self, "max_iterations", max_iterations)
        object.__setattr__(
            self, "zero_average_xc_potential", bool(self.zero_average_xc_potential)
        )


@dataclass(frozen=True)
class InitialDensitySettings:
    """Select and configure the valence-density SCF initial guess.

    ``sad`` is the PARSEC-compatible superposition of atomic densities and is
    the unchanged default.  ``file`` imports a portable ``.npz`` field or a
    legacy three-dimensional ``.npy`` field.  ``charge3net`` and ``scdp`` run
    an isolated model environment through the optional :mod:`MLDensity`
    adapters and cache their prediction as the same portable ``.npz`` format;
    ``custom`` runs the script ``bridge`` the same way, and further models can
    be registered in :mod:`MLDensity.registry`.  Supplying ``file`` with a
    named provider imports an earlier prediction from that provider without
    launching its environment.

    The DFT grid always remains authoritative.  A model prediction is either
    evaluated at those exact active grid points or interpolated onto them; it
    is never allowed to silently change ``GridSettings``.
    """

    method: InitialDensityMethod = "sad"
    file: str | Path | None = None
    units: DensityUnits = "auto"
    negative_policy: DensityNegativePolicy = "clip"
    interpolation: Literal["linear", "nearest"] = "linear"
    repository: str | Path | None = None
    checkpoint: str | Path | None = None
    python_executable: str | Path | None = None
    model: str = "qm9"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    cache_directory: str | Path | None = None
    regenerate: bool = False
    prediction_chunk_size: int = 50000
    # The script of ``method="custom"``.  It enters the reference cache key
    # of the accelerated driver through the provider fingerprint (its path
    # and bytes), so it stays out of the printed settings, which the stored
    # records of earlier trees compare.
    bridge: str | Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        method = str(self.method).strip().lower()
        if method == "ml":
            method = "file"
        provider = None
        if method not in {"sad", "file"}:
            # Imported here: a SAD calculation never loads the ML package.
            from .MLDensity.registry import get_provider, initial_density_methods

            try:
                provider = get_provider(method)
            except ValueError:
                raise ValueError(
                    "initial density method must be one of "
                    + ", ".join(initial_density_methods())
                ) from None
        if self.units not in {
            "auto",
            "e_per_bohr3",
            "e_per_angstrom3",
            "electrons_per_voxel",
        }:
            raise ValueError("unsupported initial-density units")
        if self.negative_policy not in {"clip", "error", "allow"}:
            raise ValueError("negative_policy must be clip, error, or allow")
        if self.interpolation not in {"linear", "nearest"}:
            raise ValueError("initial-density interpolation must be linear or nearest")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("ML density device must be auto, cpu, or cuda")
        chunk_size = int(self.prediction_chunk_size)
        if chunk_size != self.prediction_chunk_size or chunk_size < 1:
            raise ValueError("prediction_chunk_size must be a positive integer")
        if method == "file" and self.file is None:
            raise ValueError("initial density method 'file' requires ML_Density_File")
        if method == "sad" and self.file is not None:
            raise ValueError("ML_Density_File cannot be combined with Initial_Density=sad")
        model = str(self.model).strip().lower()
        if not model:
            raise ValueError("ML density model name cannot be empty")
        if provider is not None:
            model = provider.check_model(model)
            if provider.bridge is None and self.bridge is None and self.file is None:
                raise ValueError(
                    f"Initial_Density: {method} needs ML_Density_Bridge, the "
                    "script that runs the model (or ML_Density_File)"
                )
        elif self.bridge is not None:
            raise ValueError("ML_Density_Bridge needs a model Initial_Density")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "prediction_chunk_size", chunk_size)
        for name in (
            "file",
            "repository",
            "checkpoint",
            "python_executable",
            "cache_directory",
            "bridge",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


#: ``Max_Step`` default of ``usrinputfile.F90``.  BFGS runs unrestricted
#: when the input leaves ``Max_Step`` at this value.
DEFAULT_RELAXATION_MAXIMUM_STEP = 0.05

RelaxationMethod = Literal["simple", "bfgs", "manual"]


@dataclass(frozen=True)
class RelaxationSettings:
    """Input of a structural relaxation (``usrinputfile.F90``).

    ``method`` is ``Minimization``: ``simple`` (steepest descent), ``bfgs``
    (limited-memory BFGS) or ``manual`` (structures read from
    ``manual.dat``).  ``max_movements`` is ``Movement_Num``;
    ``minimum_force`` is ``Force_Min`` in Ry/bohr; ``maximum_step`` and
    ``minimum_step`` are ``Max_Step`` and ``Min_Step`` in bohr;
    ``bfgs_corrections`` is ``BFGS_number_corr``; ``restart`` is
    ``Relax_Restart``.  ``scf_upscale`` is ``Relaxation_SCF_Upscale``, not
    a PARSEC keyword: how much looser than ``Convergence_Criterion`` the SCF
    loop of a structure may be while the forces are large (1, the default,
    keeps the criterion fixed as PARSEC does; see
    :func:`parsec_python.Relaxation.ionic_loop.adaptive_scf_criterion`).
    ``write_restart`` is PARSEC 1.5's ``Write_Relax_Restart``: whether
    ``relax_restart.dat`` is written after every move.  1.5 writes it only
    on request (``relaxation/domove.f90:390``, "I think this is broken");
    here it is written unless the input says ``.false.`` (the writer works
    and is atomic).
    """

    method: RelaxationMethod
    max_movements: int = 500
    minimum_force: float = 1.0e-2
    maximum_step: float = DEFAULT_RELAXATION_MAXIMUM_STEP
    minimum_step: float = 1.0e-4
    bfgs_corrections: int = 7
    restart: bool = False
    scf_upscale: float = 1.0
    write_restart: bool = True

    def __repr__(self) -> str:
        # Write_Relax_Restart is new; the repr of a run without it is unchanged.
        return _repr_without_defaults(self, {"write_restart": True})

    def __post_init__(self) -> None:
        if self.method not in {"simple", "bfgs", "manual"}:
            raise ValueError(f"unknown minimization method: {self.method!r}")
        if int(self.max_movements) != self.max_movements or self.max_movements < 0:
            raise ValueError("Movement_Num must be a nonnegative integer")
        if not self.minimum_force >= 1.0e-5:
            raise ValueError(
                "Force_Min is below numerical accuracy; increase it to at least 1e-5"
            )
        if self.method == "simple":
            if not self.maximum_step >= 1.0e-5:
                raise ValueError(
                    "Max_Step is too small; increase it to at least 1e-5"
                )
            # domove.f90 takes log((stepmax-stepmin)/(stepmax-50*stepmin)):
            # outside this window the Fortran moves the atoms by NaN.
            if not self.maximum_step > 50.0 * self.minimum_step > 0.0:
                raise ValueError(
                    "the simple minimization needs Max_Step > 50*Min_Step > 0"
                )
        if self.method == "bfgs" and self.bfgs_corrections < 1:
            raise ValueError("BFGS_number_corr must be positive")
        if not (np.isfinite(self.scf_upscale) and self.scf_upscale >= 1.0):
            raise ValueError("Relaxation_SCF_Upscale must be a number of at least 1")
        object.__setattr__(self, "scf_upscale", float(self.scf_upscale))
        object.__setattr__(self, "max_movements", int(self.max_movements))
        object.__setattr__(self, "bfgs_corrections", int(self.bfgs_corrections))

    @property
    def force_threshold(self) -> float:
        """Force below which the structure counts as relaxed; zero for manual."""

        return 0.0 if self.method == "manual" else float(self.minimum_force)

    @property
    def bfgs_bound(self) -> float:
        """Largest displacement of a coordinate from its start, or ``-1``.

        ``usrinputfile.F90`` compares ``Max_Step`` with its default: an input
        that leaves it there (or sets exactly that value) runs unrestricted.
        """

        if self.maximum_step == DEFAULT_RELAXATION_MAXIMUM_STEP:
            return -1.0
        return float(self.maximum_step)


@dataclass(frozen=True)
class BandSegment:
    """One straight piece of a band path (Fortran ``bandstruc`` block line).

    ``start``/``end`` are in *relative* coordinates, i.e. multiples of the
    reciprocal lattice vectors; ``name`` is the segment label (``gamma-X``).
    """

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    name: str = ""

    def __post_init__(self) -> None:
        for field_name in ("start", "end"):
            values = tuple(float(value) for value in getattr(self, field_name))
            if len(values) != 3 or not all(np.isfinite(values)):
                raise ValueError(f"band segment {field_name} must be three finite numbers")
            object.__setattr__(self, field_name, values)
        if self.start == self.end:
            raise ValueError("a band segment must have non-zero length")


@dataclass(frozen=True)
class BandStructureSettings:
    """Fortran ``bandstruc`` / ``bandstruc_points``: a band structure computed
    after the SCF along a path of segments.

    The shortest segment gets ``points`` k-points and the others
    proportionally more (see :func:`~parsec_python.Grid.kpoints.band_path_kpoints`).
    ``rounding`` is how a segment's share is made an integer: ``"truncate"``
    (this port's rule, that of the previous Fortran tree) or ``"nint"``
    (PARSEC 1.5, ``utilities/calc_bands.f90:152-158``).  This port keeps its
    default, 20 points with truncation, where the input gives no
    ``Bandstruc_Points`` (PARSEC 1.5's default is 100); a written
    ``Bandstruc_Points`` takes 1.5's rounding (since 2026-11-04 the rule for
    defaults that 1.5 changed).  Each k-point is solved from scratch
    with the SCF's first eigensolver (CHEBFF: ``FF_MaxIter`` filter cycles of
    degree ``Chebdav_Degree``) in the converged self-consistent potential,
    exactly as Fortran ``calc_bands``.  ``ff_maxiter`` is 1.5's
    ``Bands_FF_MaxIter``: the CHEBFF filter cycles of the band k-points
    (``calc_bands.f90:336-338``); ``None`` keeps ``FF_MaxIter``.
    """

    segments: tuple[BandSegment, ...]
    points: int = 20
    ff_maxiter: int | None = None
    rounding: str = "truncate"

    def __repr__(self) -> str:
        # Bands_FF_MaxIter and the rounding are new; the repr of a run without them is unchanged.
        return _repr_without_defaults(self, {"ff_maxiter": None, "rounding": "truncate"})

    def __post_init__(self) -> None:
        segments = tuple(
            segment if isinstance(segment, BandSegment) else BandSegment(**segment)
            for segment in self.segments
        )
        if not segments:
            raise ValueError("a band structure needs at least one segment")
        object.__setattr__(self, "segments", segments)
        if int(self.points) < 1:
            raise ValueError("bandstruc_points must be at least 1")
        object.__setattr__(self, "points", int(self.points))
        if self.ff_maxiter is not None:
            if int(self.ff_maxiter) != self.ff_maxiter or int(self.ff_maxiter) < 1:
                raise ValueError("bands_ff_maxiter must be a positive integer")
            object.__setattr__(self, "ff_maxiter", int(self.ff_maxiter))
        if self.rounding not in ("truncate", "nint"):
            raise ValueError("the rounding of band-path points is truncate or nint")


#: Values of ``Output_Grid_Data`` (``usrinputfile.F90`` l.2598-2638), in the
#: order of ``all``; each names one cube file of ``export_grid_data.f90``.
GRID_DATA_FIELDS = ("charge_density", "v_external", "v_hartree", "v_external+v_hartree")


@dataclass(frozen=True)
class WavefunctionSelection:
    """Which wavefunctions ``parsec.dat`` holds (``elec_st%nsave``/``indxsave``).

    ``mode`` is ``"occupied"`` (no ``Save_Wfn_Bands`` block: ``nsave = -1``,
    states ``1..ifmax`` of each spin channel), ``"all"``
    (``Output_All_States``: every computed state) or ``"list"`` (the 1-based
    state numbers of a ``Save_Wfn_Bands`` block; an empty block gives none).
    ``explicit`` says that the input chose it; the default choice may be
    narrowed to no wavefunctions where the file would be very large (see
    :mod:`parsec_python.Output.output_files`).
    """

    mode: Literal["occupied", "all", "list"] = "occupied"
    indices: tuple[int, ...] = ()
    explicit: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"occupied", "all", "list"}:
            raise ValueError("wavefunction selection must be occupied, all or list")
        indices = tuple(int(value) for value in self.indices)
        if self.mode != "list" and indices:
            raise ValueError("only a list selection holds state numbers")
        if any(value < 1 for value in indices):
            raise ValueError("Save_Wfn_Bands state numbers start at 1")
        object.__setattr__(self, "indices", indices)


@dataclass(frozen=True)
class OutputFileSettings:
    """The output files of ``usrinputfile.F90`` l.317-318 and 2413-2666.

    ``output_data`` (``Output_Data``, default true) gates ``parsec.dat`` and
    ``eigen.dat``; ``wavefunctions`` selects the wavefunctions of
    ``parsec.dat``; ``intermediate_density``
    (``Save_Intermediate_Charge_Density``) writes ``parsec.dat`` after every
    SCF iteration.  ``eigenvalues`` (``Output_Eigenvalues``, forced on with
    k-points) writes ``eigen.dat``, with every state when ``all_states``
    (``Output_All_States``) and one block per SCF iteration when
    ``intermediate_eigenvalues`` (``Save_Intermediate_Eigenvalues``).
    ``grid_data`` lists the fields of ``Output_Grid_Data`` in the order of
    the input (values of :data:`GRID_DATA_FIELDS`), ``output_file_name``
    is ``Output_File_Name``.  ``keywords`` holds the names of these keywords
    that the input gives, so that a run that gives none reports nothing new.

    These settings describe files only; they are not part of the problem
    and change no number of the calculation.
    """

    output_data: bool = True
    wavefunctions: WavefunctionSelection = field(default_factory=WavefunctionSelection)
    intermediate_density: bool = False
    eigenvalues: bool = False
    all_states: bool = False
    intermediate_eigenvalues: bool = False
    grid_data: tuple[str, ...] = ()
    output_file_name: str = "parsec.out"
    keywords: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        unknown = [name for name in self.grid_data if name not in GRID_DATA_FIELDS]
        if unknown:
            raise ValueError(f"unknown Output_Grid_Data field(s): {', '.join(unknown)}")
        name = str(self.output_file_name).strip()
        if not name:
            raise ValueError("Output_File_Name cannot be empty")
        object.__setattr__(self, "output_file_name", name)
        object.__setattr__(self, "grid_data", tuple(self.grid_data))
        object.__setattr__(self, "keywords", frozenset(self.keywords))

    @property
    def writes_parsec_dat(self) -> bool:
        return bool(self.output_data)

    @property
    def writes_eigen_dat(self) -> bool:
        return bool(self.output_data and self.eigenvalues)


@dataclass(frozen=True)
class NSCFSettings:
    """A non-self-consistent run (Fortran ``calc_nscf.F90``, ``nscf_*`` keys).

    ``k_points`` are Cartesian (1/bohr, one per row) after ``nscf_kpoint_unit``
    (reciprocal lattice vectors of the input cell, as ``usrinputfile.F90``
    l.1071-1085 converts them), ``weights`` the ``nscf_kpoints`` weights
    renormalized to sum 1, ``number_of_states`` is ``nscf_states_num``
    (default ``States_Num``).  The potential comes from ``parsec.dat``
    (``Restart_Run``); no SCF loop runs.  These settings are not part of the
    problem: the SCF problem of the input stays what it is.
    """

    k_points: np.ndarray
    weights: np.ndarray
    number_of_states: int

    def __post_init__(self) -> None:
        k_points = np.asarray(self.k_points, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if k_points.ndim != 2 or k_points.shape[1] != 3 or k_points.shape[0] < 1:
            raise ValueError("nscf k_points must have shape (n, 3) with n >= 1")
        if weights.shape != (k_points.shape[0],) or not np.all(np.isfinite(weights)):
            raise ValueError("nscf weights must hold one finite value per k-point")
        if int(self.number_of_states) != self.number_of_states or self.number_of_states < 1:
            raise ValueError("nscf_states_num must be a positive integer")
        object.__setattr__(self, "k_points", k_points)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "number_of_states", int(self.number_of_states))


@dataclass(frozen=True)
class DOSSettings:
    """``Create_DOS`` and its keywords (``usrinputfile.F90`` l.1122-1197, 1913).

    ``points`` is ``DOS_Pnum`` (energy mesh points).  For a periodic system
    ``monkhorst_pack_grid`` is the grid of the tetrahedra (``DOS_MP_Grid``,
    else the SCF grid) and ``monkhorst_pack_shift`` its ``DOS_MP_Shift``
    (``None`` without the block).  ``ylm_l`` is ``YLM_DOS_L`` (``-1``: no
    projected DOS; ``0..3``: the real harmonics s to f) and
    ``ylm_cutoffs`` the per-species ``ylm_dos_cutoff`` in bohr (default 1),
    as ``(symbol, radius)`` pairs in the order of the input.
    ``fortran_projection`` reproduces the projection of ``ylm_eig`` with its
    two defects (an s channel without the radial Gaussian and no Bloch phase
    of the image cells; a Python extension switch, see
    :mod:`parsec_python.Output.dos`).  A report setting: not part of the
    problem.
    """

    points: int = 1000
    monkhorst_pack_grid: tuple[int, int, int] | None = None
    monkhorst_pack_shift: tuple[float, float, float] | None = None
    ylm_l: int = -1
    ylm_cutoffs: tuple[tuple[str, float], ...] = ()
    fortran_projection: bool = False

    def __post_init__(self) -> None:
        if int(self.points) != self.points or self.points < 1:
            raise ValueError("DOS_Pnum must be a positive integer")
        if not -1 <= int(self.ylm_l) <= 3:
            raise ValueError("YLM_DOS_L must be -1 (off) or 0, 1, 2, 3")
        object.__setattr__(self, "points", int(self.points))
        object.__setattr__(self, "ylm_l", int(self.ylm_l))
        object.__setattr__(
            self,
            "ylm_cutoffs",
            tuple((str(symbol), float(radius)) for symbol, radius in self.ylm_cutoffs),
        )


@dataclass(frozen=True)
class GWSettings:
    """``Output_GW``: the WFN, RHO and VXC files of BerkeleyGW (``gw_write.F90``).

    ``space_group`` is the :class:`parsec_python.Grid.space_group.SpaceGroup`
    of the files (PARSEC's ``symm``: the full group, the identity with
    ``Ignore_Symmetry``).  ``kgrid``/``kshift`` go to the WFN header: with a
    non-self-consistent run ``nscf_kgrid``/``nscf_kgrid_shift``
    (``parsec.F90`` l.2133-2134), with an SCF the Monkhorst-Pack grid of the
    SCF (see :mod:`parsec_python.Output.berkeleygw`).  ``fermi_level`` is
    ``nscf_fermi_level`` (Ry, the fixed Fermi level of the NSCF occupations),
    ``None`` with an SCF.  A report setting: not part of the problem.
    """

    space_group: object
    kgrid: tuple[int, int, int] = (0, 0, 0)
    kshift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fermi_level: float | None = None

    def __post_init__(self) -> None:
        kgrid = tuple(int(value) for value in self.kgrid)
        kshift = tuple(float(value) for value in self.kshift)
        if len(kgrid) != 3 or len(kshift) != 3:
            raise ValueError("the GW k-grid and its shift need three entries each")
        object.__setattr__(self, "kgrid", kgrid)
        object.__setattr__(self, "kshift", kshift)
        if self.fermi_level is not None:
            object.__setattr__(self, "fermi_level", float(self.fermi_level))


# --- FDET (PARSEC 1.5 frozen-density embedding) --------------------------------


@dataclass(frozen=True)
class EmbeddingSettings:
    """``Potential_Field``: frozen-density embedding of PARSEC 1.5 (FDET/AFM).

    The run is the active tip ``B`` in the frozen field of a sample ``A``
    that an earlier run wrote to ``field_path`` (``Potential_Field_Name``,
    a ``pot.dat`` of :mod:`parsec_python.Output.pot_dat`).
    ``kinetic_functional`` is ``Kinetic_Energy_Functional`` (``tf``, ``tw``,
    ``pw``, ``pb``, ``none``), ``pbe_mu``/``pbe_kappa`` are ``pbe_m``/``pbe_k``
    of ``pb``, ``kinetic_form`` is ``FDET_Kinetic_Form`` (``corrected``, the
    default, or ``parsec1.5``, :mod:`parsec_python.Embedding.kinetic`), and
    ``tip_sphere`` is ``Boundary_Sphere_Around_Tip`` (the tip in a sphere of
    its own, :mod:`parsec_python.Embedding.tip_sphere`).
    """

    field_path: Path
    kinetic_functional: str = "tf"
    pbe_mu: float = 0.2319
    pbe_kappa: float = 0.8438
    kinetic_form: str = "corrected"
    tip_sphere: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_path", Path(self.field_path))
        if self.kinetic_functional not in ("tf", "tw", "pw", "pb", "none"):
            raise ValueError("kinetic_functional must be tf, tw, pw, pb or none")
        if self.kinetic_form not in ("corrected", "parsec1.5"):
            raise ValueError("kinetic_form must be corrected or parsec1.5")
        object.__setattr__(self, "pbe_mu", float(self.pbe_mu))
        object.__setattr__(self, "pbe_kappa", float(self.pbe_kappa))
        object.__setattr__(self, "tip_sphere", bool(self.tip_sphere))


@dataclass(frozen=True)
class SinglePointInput:
    """
    Complete input to the modular single-point calculator.
    """

    atoms: Sequence[Atom]
    pseudopotentials: Mapping[str, SpeciesPotential]
    grid: GridSettings | PeriodicGridSettings
    periodic_cell: PeriodicCell | None = None
    scf: SCFSettings = field(default_factory=SCFSettings)
    hartree: HartreeSettings = field(default_factory=HartreeSettings)
    eigensolver: EigensolverSettings = field(default_factory=EigensolverSettings)
    mixing: MixingSettings = field(default_factory=MixingSettings)
    initial_density_settings: InitialDensitySettings = field(
        default_factory=InitialDensitySettings
    )
    recenter_geometry: bool = True
    monkhorst_pack_dimensions: tuple[int, int, int] | None = None
    """PARSEC's ``Kpoint_Method: mp`` + ``Monkhorst_Pack_Grid`` block: an
    unreduced, uniform (equal-weight) Brillouin-zone sample, implemented by
    :func:`~parsec_python.SCF.kpoints.run_scf_kpoints` for periodic,
    spin-unpolarized, non-SOC inputs.  ``None`` (the default) is the
    ordinary Gamma-only periodic path."""
    kpoints: np.ndarray | None = None
    """An explicit k-point list (Fortran ``Kpoint_Method: manual``), shape
    ``(n, 3)`` in Cartesian 1/bohr; mutually exclusive with
    ``monkhorst_pack_dimensions`` and requiring ``periodic_cell``.  Convert
    user units with :func:`~parsec_python.Grid.kpoints.explicit_kpoints`."""
    kpoint_weights: np.ndarray | None = None
    """Sampling weights for ``kpoints`` (non-negative, normalized to sum 1 as
    PARSEC's ``init_var`` does; equal if omitted).  Unequal weights use the
    per-state weights of the Fermi level (Fortran ``flevel``)."""
    bands: BandStructureSettings | None = None
    """Non-self-consistent band structure to compute after the SCF (periodic
    systems only); see :class:`BandStructureSettings`."""
    point_charges: tuple[PointCharge, ...] = ()
    """Fixed classical point charges of an isolated system (PARSEC's
    ``Add_Point_Charges``), at the coordinates of the input: they are
    recentred with the atoms where ``recenter_geometry`` moves them."""
    charged_sheets: tuple[ChargedSheet, ...] = ()
    """Charged planes of a slab (PARSEC's ``add_charged_sheets``), in the
    order of the input."""
    monkhorst_pack_shift: tuple[float, float, float] | None = None
    """PARSEC's ``Monkhorst_Pack_Shift`` block.  Given, the grid of
    ``monkhorst_pack_dimensions`` is Fortran's ``k = ((i-1) + s_i)/n_i`` and
    is reduced to the irreducible wedge by ``space_group`` and time reversal
    (:func:`~parsec_python.Grid.kpoints.irreducible_monkhorst_pack`,
    ``irrbz.f90``).  ``None``: this port's unreduced grid, as before."""
    space_group: object | None = None
    """The :class:`~parsec_python.Grid.space_group.SpaceGroup` of a periodic
    cell (Fortran ``symm``), found by the input reader where the
    calculation uses it: the irreducible wedge of a ``monkhorst_pack_shift``
    grid and ``SCFSettings.symmetrize_density``.  ``None`` stands for the
    identity alone."""
    embedding: EmbeddingSettings | None = None
    """PARSEC 1.5's frozen-density embedding (``Potential_Field``) of an
    isolated, spin-unpolarized tip run; ``None`` without it."""

    def __repr__(self) -> str:
        # Point charges and charged sheets that are not there, and the
        # k-point shift and space group where they are not given, print
        # nothing, so that the text of every calculation without them
        # stays what it was.
        hidden = {
            "point_charges": not self.point_charges,
            "charged_sheets": not self.charged_sheets,
            "monkhorst_pack_shift": self.monkhorst_pack_shift is None,
            "space_group": self.space_group is None,
            "embedding": self.embedding is None,
        }
        shown = [
            f"{item.name}={getattr(self, item.name)!r}"
            for item in fields(self)
            if not hidden.get(item.name, False)
        ]
        return f"{type(self).__qualname__}({', '.join(shown)})"

    @property
    def has_kpoint_sampling(self) -> bool:
        """Whether k-points beyond Gamma are requested (grid or explicit list)."""
        return self.monkhorst_pack_dimensions is not None or self.kpoints is not None

    def __post_init__(self) -> None:
        atoms = tuple(atom if isinstance(atom, Atom) else Atom(**atom) for atom in self.atoms)
        if not atoms:
            raise ValueError("a single-point calculation requires at least one atom")
        missing = sorted({atom.symbol for atom in atoms}.difference(self.pseudopotentials))
        if missing:
            raise ValueError(f"missing pseudopotential specifications for: {', '.join(missing)}")
        object.__setattr__(self, "atoms", atoms)
        if self.periodic_cell is None:
            if not isinstance(self.grid, GridSettings):
                raise ValueError(
                    "an isolated calculation (periodic_cell=None) requires GridSettings"
                )
        elif not isinstance(self.grid, PeriodicGridSettings):
            raise ValueError(
                "a periodic calculation (periodic_cell set) requires PeriodicGridSettings"
            )
        if self.monkhorst_pack_dimensions is not None:
            if self.periodic_cell is None:
                raise ValueError("monkhorst_pack_dimensions requires periodic_cell")
            dims = tuple(int(value) for value in self.monkhorst_pack_dimensions)
            if len(dims) != 3 or any(value < 1 for value in dims):
                raise ValueError("monkhorst_pack_dimensions must be three positive integers")
            object.__setattr__(self, "monkhorst_pack_dimensions", dims)
        if self.monkhorst_pack_shift is not None:
            if self.monkhorst_pack_dimensions is None:
                raise ValueError("monkhorst_pack_shift requires monkhorst_pack_dimensions")
            shift = tuple(float(value) for value in self.monkhorst_pack_shift)
            if len(shift) != 3 or any(
                not np.isfinite(value) or value < 0.0 for value in shift
            ):
                # usrinputfile.F90 error 117.
                raise ValueError("monkhorst_pack_shift must be three non-negative numbers")
            object.__setattr__(self, "monkhorst_pack_shift", shift)
        if self.space_group is not None and self.periodic_cell is None:
            raise ValueError("space_group belongs to a periodic cell")
        if self.kpoints is not None:
            if self.periodic_cell is None:
                raise ValueError("an explicit k-point list requires periodic_cell")
            if self.monkhorst_pack_dimensions is not None:
                raise ValueError(
                    "give either monkhorst_pack_dimensions or an explicit kpoints "
                    "list, not both"
                )
            k_points = np.asarray(self.kpoints, dtype=float)
            if k_points.ndim != 2 or k_points.shape[0] < 1 or k_points.shape[1] != 3:
                raise ValueError("kpoints must have shape (n, 3) with n >= 1")
            if not np.all(np.isfinite(k_points)):
                raise ValueError("kpoints must be finite")
            if self.kpoint_weights is None:
                weights = np.full(k_points.shape[0], 1.0 / k_points.shape[0])
            else:
                weights = np.asarray(self.kpoint_weights, dtype=float)
                if weights.shape != (k_points.shape[0],) or not np.all(np.isfinite(weights)):
                    raise ValueError("kpoint_weights must have one finite value per k-point")
                if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
                    raise ValueError("kpoint_weights must be non-negative and not all zero")
                weights = weights / float(np.sum(weights))
            object.__setattr__(self, "kpoints", k_points)
            object.__setattr__(self, "kpoint_weights", weights)
        elif self.kpoint_weights is not None:
            raise ValueError("kpoint_weights given without kpoints")
        if self.bands is not None and self.periodic_cell is None:
            raise ValueError("a band structure requires periodic_cell")
        charges = tuple(
            item if isinstance(item, PointCharge) else PointCharge(**item)
            for item in self.point_charges
        )
        if charges and self.periodic_cell is not None:
            # usrinputfile.F90 stops with error 145.
            raise ValueError("point charges are supported for isolated systems only")
        object.__setattr__(self, "point_charges", charges)
        sheets = tuple(
            item if isinstance(item, ChargedSheet) else ChargedSheet(**item)
            for item in self.charged_sheets
        )
        if sheets and getattr(self.grid, "periodicity", None) != 2:
            # usrinputfile.F90 stops with error 145.
            raise ValueError("charged sheets are supported for slab systems only")
        object.__setattr__(self, "charged_sheets", sheets)


@dataclass
class BandStructureResult:
    """Eigenvalues along a band path computed with a fixed SCF potential.

    ``eigenvalues`` has shape ``(n_channels, n_kpoints, n_states)`` in Ry,
    with ``channels`` naming the first axis: ``("all",)`` for scalar and
    spinor Hamiltonians, ``("up", "down")`` for collinear spin-polarized runs
    without spin-orbit.  ``k_points`` are Cartesian (1/bohr) and flattened
    over the segments; ``segment_counts[i]`` consecutive k-points belong to
    segment ``i`` (``segment_names[i]``).  ``fermi_level`` is the SCF Fermi
    level in the same energy zero.  ``residual_norms`` (channel, k) is the
    largest ``|H v - e v|`` over the reported states: the Fortran CHEBFF
    solver has no convergence test, so this is the quality diagnostic.
    ``spin_expectation`` (spinor Hamiltonians) is ``<sigma_x,y,z>`` of every
    state, shape ``(1, n_kpoints, n_states, 3)``.
    """

    mode: str
    channels: tuple[str, ...]
    segment_names: tuple[str, ...]
    segment_counts: np.ndarray
    k_points: np.ndarray
    eigenvalues: np.ndarray
    residual_norms: np.ndarray
    fermi_level: float
    spin_expectation: np.ndarray | None = None
    attempts: np.ndarray | None = None
    """Diagonalization attempts per ``(channel, k-point)``: 1 unless the CHEBDAV
    re-solve with a larger block was needed (see :mod:`parsec_python.SCF.bands`)."""

    @property
    def segment_index(self) -> np.ndarray:
        """Segment number (0-based) of every k-point."""
        return np.repeat(np.arange(len(self.segment_counts)), self.segment_counts)

    @property
    def path_distance(self) -> np.ndarray:
        """Cumulative Cartesian arc length along the path (1/bohr)."""
        steps = np.linalg.norm(np.diff(self.k_points, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(steps)])


@dataclass(frozen=True)
class EnergyBreakdown:
    """PARSEC ``totnrg`` terms, all in Rydberg."""

    eigenvalue: float
    hartree: float
    integral_vxc_rho: float
    exchange_correlation: float
    electron_ion: float
    ion_ion: float
    electronic: float
    total: float


@dataclass(frozen=True)
class CorrectedEnergyBreakdown(EnergyBreakdown):
    """:class:`EnergyBreakdown` with the LDA+U and dispersion terms, in Ry.

    ``hubbard`` is ``etot_plusu`` of ``calcden.f90z``, which ``totnrg.f90``
    adds to the electronic energy: it is part of ``electronic`` and of
    ``total``.  ``dispersion`` is the Tkatchenko-Scheffler energy of
    ``efvdw.f90``, which the report after convergence adds to ``total``
    only.  A calculation with neither keeps the plain
    :class:`EnergyBreakdown`, whose fields (and archive members) stay what
    they were.
    """

    hubbard: float = 0.0
    dispersion: float = 0.0


@dataclass(frozen=True)
class HybridEnergyBreakdown(EnergyBreakdown):
    """:class:`EnergyBreakdown` of a hybrid functional (generalized Kohn-Sham), in Ry.

    ``eigenvalue`` is the band energy of the Hamiltonian with the exchange
    operator ``a_x V_x``; ``exact_exchange_band`` is the part of it that
    operator gives, ``w sum_i f_i <q_i|a_x V_x|q_i>``, which ``electronic``
    removes (double counting).  ``exact_exchange`` is the scaled exact
    exchange energy ``a_x E_x`` the total holds and
    ``semilocal_exchange_correlation`` the semilocal part of the functional
    (with its reduced semilocal exchange); ``exchange_correlation`` is their
    sum.  ``integral_vxc_rho`` is that of the semilocal potential.

    Within an inner SCF loop the operator is fixed (built from the orbitals
    ``P0`` of the outer iteration before) and ``linearized`` is true:
    ``exact_exchange = a_x (2 E_x(P, P0) - E_x(P0))``, the exchange energy of
    the functional that loop minimizes.  After the loop ``exact_exchange``
    is ``a_x E_x(P)`` of the new orbitals and ``linearized`` false.
    """

    semilocal_exchange_correlation: float = 0.0
    exact_exchange: float = 0.0
    exact_exchange_band: float = 0.0
    exact_exchange_fraction: float = 0.0
    linearized: bool = False


@dataclass(frozen=True)
class ExchangeLoopIteration:
    """One outer iteration of the Fock loop of a hybrid SCF (energies in Ry).

    ``iteration`` 0 is the operator built from the orbitals of the start
    (the semilocal SCF ``start_functional``, or the structure before a move);
    iteration ``n >= 1`` reports the inner SCF loop run with the operator of
    iteration ``n - 1``.  ``exchange_energy`` is ``E_x`` of the orbitals of
    this iteration (no fraction), ``cross_exchange_energy`` the exchange
    energy of those orbitals with the operator they were solved with,
    ``E_x(P_n, P_n-1)``, ``estimated_error`` Quantum ESPRESSO's ``dexx``,
    ``a_x (2 E_x(P_n, P_n-1) - E_x(P_n-1) - E_x(P_n))`` (nonnegative), and
    ``total_energy`` the total energy of the hybrid functional of the
    orbitals of this iteration.  ``pair_solves`` and the times are those of
    this iteration's exchange evaluation; ``operator_ranks`` the number of
    orbitals of the operator built after it, per spin channel (empty when no
    operator was built).
    """

    iteration: int
    functional: str
    fraction: float
    exchange_energy: float
    total_energy: float | None = None
    previous_exchange_energy: float | None = None
    cross_exchange_energy: float | None = None
    estimated_error: float | None = None
    tolerance: float = 0.0
    inner_iterations: int = 0
    inner_converged: bool = True
    converged: bool = False
    pair_solves: int = 0
    poisson_seconds: float = 0.0
    exchange_seconds: float = 0.0
    operator_ranks: tuple[int, ...] = ()
    start_functional: str | None = None
    pair_solver: str = ""
    operator_asymmetry: float = 0.0


@dataclass(frozen=True)
class EmbeddingEnergyBreakdown(EnergyBreakdown):
    """:class:`EnergyBreakdown` of an FDET tip run (``totnrg.f90:159-197``), in Ry.

    ``electron_ion`` reads the ionic potential with the sample field
    (Fortran's ``eion``), ``exchange_correlation`` is
    ``E_xc[rho_B + rho_core + rho_A]`` and ``integral_vxc_rho`` its potential
    times ``rho_B``.  The other terms: ``nonadditive_kinetic`` ``T_nad``,
    ``integral_vnad_rho`` ``Integral v_nad rho_B`` (it cancels in the total),
    ``tip_ion_sample_electron`` ``Integral rho_A v_tip,ion``, ``sample_hartree``
    ``E_H[rho_A] = Integral rho_A v_H^A / 2``, ``sample_electron_ion``
    ``Integral rho_A v_A,ion``, ``sample_kinetic`` the ``newk`` of the sample
    file, and with the tip sphere ``sample_xc_on_tip_grid`` (``E_xc[rho_A]``
    on the tip grid) and ``tip_ion_sample_electron_outside`` (the
    attraction of the sample electrons outside the tip sphere).  With the tip
    sphere the total leaves out the sample Hartree, electron-ion, kinetic and
    exchange-correlation energies, a constant of the sample.  ``electronic``
    holds every term but ``ion_ion``.
    """

    nonadditive_kinetic: float = 0.0
    integral_vnad_rho: float = 0.0
    tip_ion_sample_electron: float = 0.0
    sample_hartree: float = 0.0
    sample_electron_ion: float = 0.0
    sample_kinetic: float = 0.0
    tip_sphere: bool = False
    sample_xc_on_tip_grid: float = 0.0
    tip_ion_sample_electron_outside: float = 0.0


@dataclass(frozen=True)
class PreparationTimings:
    """Wall-clock timings for density-independent single-point preparation.

    Every field has a zero default so adding timing metadata does not make
    callers construct it explicitly.  Times are measured with
    :func:`time.perf_counter` and are expressed in seconds.
    """

    pseudopotential_loading_seconds: float = 0.0
    grid_seconds: float = 0.0
    finite_difference_seconds: float = 0.0
    local_ionic_seconds: float = 0.0
    nonlocal_ionic_seconds: float = 0.0
    initial_density_seconds: float = 0.0
    core_density_seconds: float = 0.0
    ion_ion_seconds: float = 0.0
    # Estimate of the Hartree boundary and, where one is built, its tail.
    hartree_boundary_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(frozen=True)
class RunTimings:
    """Aggregated wall-clock timings for one SCF run.

    ``preparation`` carries the static setup measurements into the final
    result, while the remaining fields describe :func:`run_scf`.  Hamiltonian
    binding is deliberately separate from diagonalization: binding only joins
    the current diagonal potential to the prebuilt finite-difference and
    nonlocal operators; it does not materialize a dense Hamiltonian.
    """

    preparation: PreparationTimings = field(default_factory=PreparationTimings)
    initial_hartree_seconds: float = 0.0
    initial_xc_seconds: float = 0.0
    hamiltonian_binding_seconds: float = 0.0
    diagonalization_seconds: float = 0.0
    occupations_density_seconds: float = 0.0
    hartree_seconds: float = 0.0
    xc_seconds: float = 0.0
    mixing_energy_seconds: float = 0.0
    total_seconds: float = 0.0
    detail: dict[str, float] = field(default_factory=dict)
    """Seconds and counters of the parts of a stage that a driver measures inside its eigensolver
    workers (k-point drivers); worker seconds are summed over the workers."""


@dataclass(frozen=True)
class SCFIteration:
    """Diagnostics retained for one SCF iteration."""

    iteration: int
    weighted_residual: float
    plain_residual: float
    eigen_residual_max: float
    hartree_residual: float
    energies: EnergyBreakdown
    eigenvalues: tuple[float, ...] = ()
    occupations: tuple[float, ...] = ()
    representations: tuple[int, ...] = ()
    fermi_level: float = float("nan")
    density_minimum: float = float("nan")
    density_maximum: float = float("nan")
    diagonalization_seconds: float = 0.0
    hartree_seconds: float = 0.0
    hamiltonian_binding_seconds: float = 0.0
    occupations_density_seconds: float = 0.0
    xc_seconds: float = 0.0
    mixing_energy_seconds: float = 0.0
    total_seconds: float = 0.0
    # Spin-polarized calculations only, one entry per channel, up first.
    # ``eigenvalues``, ``occupations`` and ``representations`` then hold the
    # states of the up channel followed by those of the down channel, the
    # two residuals above are the larger of the two channels (the values
    # PARSEC tests), and the density extrema are those of the total density.
    spin_plain_residuals: tuple[float, ...] = ()
    spin_weighted_residuals: tuple[float, ...] = ()
    spin_electrons: tuple[float, ...] = ()
    spin_density_minimum: tuple[float, ...] = ()
    spin_density_maximum: tuple[float, ...] = ()
    # Symmetrize_Charge_Density only: what symm_scalar reports of each
    # density it symmetrized (parsec_python.Grid.ScalarSymmetrization), the
    # total density first and, with spin, the up and the down channel.
    density_symmetrization: tuple = ()
    # LDA+U only, from the second iteration on: the two terms
    # ``calcden.f90z`` prints before the eigenproblem of the iteration, from
    # the orbitals of the iteration before (Ry).  ``None`` otherwise.
    hubbard_first_contribution: float | None = None
    hubbard_energy: float | None = None
    # ARPACK and DIAGLA only: (converged pairs, matrix-vector products) of
    # the solve of each spin channel, as eigval.F90 reports them; each a
    # tuple of one count per representation where the accelerated solver
    # solved symmetry sectors.
    eigensolver_counts: tuple[tuple, ...] = ()
    # Dynamic_Diag_Tol only: the eigensolver tolerance of the next
    # iteration (parsec.F90:1304-1311, "New tolerance").
    next_eigensolver_tolerance: float | None = None

    @property
    def spin_polarized(self) -> bool:
        return len(self.spin_electrons) == 2

    @property
    def net_magnetic_moment(self) -> float:
        """Electrons up minus electrons down, in Bohr magnetons."""

        if not self.spin_polarized:
            return 0.0
        return float(self.spin_electrons[0] - self.spin_electrons[1])


@dataclass(frozen=True)
class SpinChannelResult:
    """One spin channel of a spin-polarized result.

    ``occupations`` are electrons per state, between zero and one.
    ``density`` is the valence density of the channel; the two channels add
    up to the density of the result.  The three effective potentials and
    ``xc_potential`` are those of the channel, with the meaning of the
    fields of the same names of :class:`SinglePointResult`.
    ``electron_count`` is the sum of the occupations (``totel`` of
    ``flevel.f90``).  ``eigensolver_state`` is the buffered eigensubspace of
    the channel.
    """

    eigenvalues: np.ndarray
    occupations: np.ndarray
    wavefunctions: np.ndarray
    representations: np.ndarray
    density: np.ndarray
    xc_potential: np.ndarray
    input_effective_potential: np.ndarray
    output_effective_potential: np.ndarray
    next_effective_potential: np.ndarray
    electron_count: float
    eigensolver_state: object | None = None


@dataclass(frozen=True)
class SpinPolarizedSCFIteration:
    """Diagnostics retained for one :func:`~parsec_python.SCF.spin_polarized.run_scf_spin_polarized`
    iteration -- the spin-polarized counterpart to :class:`SCFIteration`, with
    both channels' eigenpairs instead of one combined set."""

    iteration: int
    weighted_residual: float
    plain_residual: float
    energies: EnergyBreakdown
    eigenvalues_up: tuple[float, ...] = ()
    eigenvalues_down: tuple[float, ...] = ()
    occupations_up: tuple[float, ...] = ()
    occupations_down: tuple[float, ...] = ()
    fermi_level: float = float("nan")
    # Symmetrize_Charge_Density: the symm_scalar reports of the total, up and
    # down densities (``isp = 1, 2, 3``), empty without it.
    density_symmetrization: tuple = ()
    # LDA+U only, from the second iteration on: the two terms calcden.f90z
    # prints (Ry), as in SCFIteration; ``None`` otherwise.
    hubbard_first_contribution: float | None = None
    hubbard_energy: float | None = None
    # Dynamic_Diag_Tol only: the eigensolver tolerance of the next iteration
    # (parsec.F90:1302-1311), as in SCFIteration.
    next_eigensolver_tolerance: float | None = None
    diagonalization_seconds: float = 0.0
    total_seconds: float = 0.0
    eigen_residual_max: float = float("nan")


@dataclass
class SinglePointResult:
    """Result of a modular isolated single-point calculation.

    A spin-polarized calculation fills ``spin`` with its two channels, up
    first.  The other fields then describe both channels together:
    ``density`` is the total valence density; ``xc_potential`` and the three
    effective potentials are the averages of the two channels, which is what
    the local forces of ``forcloc.f90`` read; ``eigenvalues``,
    ``occupations``, ``representations`` and the columns of
    ``wavefunctions`` list the states of the up channel followed by those of
    the down channel; ``eigensolver_state`` holds the states of both
    channels.  The occupations are electrons per state divided by
    :attr:`spin_degeneracy`.  A backend that keeps the orbitals on a device
    leaves ``wavefunctions`` as the two blocks of
    :class:`parsec_python.SCF.spin_polarized.SpinOrbitalBlocks` until they
    are brought to the host.
    """

    converged: bool
    iterations: int
    atoms: tuple[Atom, ...]
    electron_count: float
    grid: object
    pseudopotentials: dict[str, object]
    density: np.ndarray
    core_density: np.ndarray
    ionic_potential: np.ndarray
    hartree_potential: np.ndarray
    xc_potential: np.ndarray
    input_effective_potential: np.ndarray
    output_effective_potential: np.ndarray
    next_effective_potential: np.ndarray
    nonlocal_operator: object
    eigenvalues: np.ndarray
    occupations: np.ndarray
    wavefunctions: np.ndarray
    fermi_level: float
    energies: EnergyBreakdown
    history: list[SCFIteration]
    timings: RunTimings = field(default_factory=RunTimings)
    representations: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    atomic_reference_correction: float = 0.0
    k_points: np.ndarray | None = None
    """Cartesian k-points, shape ``(n_k, 3)`` in bohr^-1, for a k-point-sampled
    run (``None`` otherwise).  Pooled eigenvalue/occupation arrays are
    k-major: entries ``i*n_states:(i+1)*n_states`` belong to ``k_points[i]``."""
    k_weights: np.ndarray | None = None
    """Normalized k-point weights, shape ``(n_k,)``."""
    # Buffered eigensubspace of the last eigenproblem.  An ionic step hands
    # it to the SCF loop of the next structure; nothing else reads it.
    eigensolver_state: object | None = None
    # The two channels of a spin-polarized calculation, up first.
    spin: tuple[SpinChannelResult, SpinChannelResult] | None = None
    # LDA+U: the :class:`parsec_python.Hamiltonian.hubbard.HubbardTerms` of the
    # last iteration (from the orbitals of the iteration before it), ``None``
    # without LDA+U or before its second iteration.
    hubbard: object | None = None
    # Van_Der_Waals: the :class:`parsec_python.Energy.dispersion.DispersionResult`
    # of the converged density, ``None`` otherwise.
    dispersion: object | None = None
    # A k-point loop asked to keep its orbitals (``keep_wavefunctions`` of
    # parsec_python.SCF.kpoints: the Ylm-projected DOS, the forces of a
    # periodic cell): the final Bloch factors ``u_nk`` of every k-point, one
    # ``(n_grid, n_states)`` array per k-point in the order of ``k_points``,
    # and the input potential of the last iteration
    # (``input_effective_potential`` of a k-point loop is the mixed potential
    # of the next one).  ``None`` otherwise.
    k_wavefunctions: tuple[np.ndarray, ...] | None = None
    scf_input_potential: np.ndarray | None = None
    nonlocal_forces: np.ndarray | None = None
    """``(atoms, 3)`` nonlocal (Kleinman-Bylander) forces in Ry/bohr, summed over the k-points from the converged
    eigenvectors of the last SCF step.  A k-point run does not keep its eigenvectors, so it accumulates this term
    while they exist (``run_scf_kpoints(..., compute_forces=True)``); ``None`` when not requested."""
    embedding: object | None = None
    """FDET (``Potential_Field``): the :class:`parsec_python.Embedding.kinetic.NonadditiveKinetic`
    of the last iteration (``T_nad`` and ``v_nad``, which ``input_effective_potential`` and
    ``output_effective_potential`` hold and ``xc_potential`` does not); ``None`` without FDET."""
    exact_exchange: object | None = None
    """A hybrid functional: the :class:`parsec_python.SCF.hybrid.FockLoopSummary` of the Fock
    loop (its outer iterations and what an ionic step needs to start the next structure);
    ``None`` otherwise.  ``xc_potential`` is then the semilocal part only."""

    @property
    def all_electron_total(self) -> float:
        """Pseudo total plus optional atomic reference corrections, in Ry."""

        return float(self.energies.total + self.atomic_reference_correction)

    @property
    def spin_polarized(self) -> bool:
        return self.spin is not None

    @property
    def spin_degeneracy(self) -> int:
        """Electrons a state with occupation one holds: two, or one with spin."""

        return 2 if self.spin is None else 1

    @property
    def net_magnetic_moment(self) -> float:
        """Electrons up minus electrons down, in Bohr magnetons."""

        if self.spin is None:
            return 0.0
        return float(self.spin[0].electron_count - self.spin[1].electron_count)

@dataclass
class SpinPolarizedSinglePointResult:
    """Result of :func:`~parsec_python.SCF.spin_polarized.run_scf_spin_polarized`.

    A deliberately narrower counterpart to :class:`SinglePointResult`: it
    carries what a caller (in particular PARSEC's ``pls.F90``-style
    perturbative spin-orbit diagonalization) needs from a converged
    collinear spin-polarized ground state -- both channels' eigenpairs, the
    shared Fermi level, and the converged energy -- without the unpolarized
    path's full per-iteration timing/history instrumentation.
    """

    converged: bool
    iterations: int
    atoms: tuple[Atom, ...]
    electron_count: float
    eigenvalues_up: np.ndarray
    eigenvalues_down: np.ndarray
    occupations_up: np.ndarray
    occupations_down: np.ndarray
    wavefunctions_up: np.ndarray
    wavefunctions_down: np.ndarray
    fermi_level: float
    density_up: np.ndarray
    density_down: np.ndarray
    energies: EnergyBreakdown
    occupation_weight: float = 1.0
    """Brillouin-zone weight carried by each pooled eigenvalue: 1 for a
    single (Gamma or cluster) calculation, ``1/n_k`` for an unreduced uniform
    k-point grid whose eigenvalues and occupations are pooled over k."""
    scf_potentials: dict[str, np.ndarray] | None = None
    """Converged local potentials of a periodic run (output side of the last
    iteration), the input of a non-self-consistent band calculation."""
    k_points: np.ndarray | None = None
    """Cartesian k-points, shape ``(n_k, 3)`` in bohr^-1, for a k-point-sampled
    run (``None`` otherwise).  Pooled eigenvalue/occupation arrays are
    k-major: entries ``i*n_states:(i+1)*n_states`` belong to ``k_points[i]``."""
    k_weights: np.ndarray | None = None
    """Normalized k-point weights, shape ``(n_k,)``."""
    k_wavefunctions: tuple[tuple[np.ndarray, ...], ...] | None = None
    """With ``keep_wavefunctions`` of the k-point driver: the final ``u_nk`` of
    every (spin channel, k-point), up first; ``None`` otherwise."""
    scf_input_potentials: tuple[np.ndarray, np.ndarray] | None = None
    """With ``keep_wavefunctions``: the (up, down) input potentials of the
    last iteration (the forces' self-consistency term); ``None`` otherwise."""
    next_potentials: tuple[np.ndarray, np.ndarray] | None = None
    """With ``keep_wavefunctions``: the (up, down) mixed potentials the next
    iteration would take (the warm start of an ionic step)."""
    xc_potentials: tuple[np.ndarray, np.ndarray] | None = None
    """With ``keep_wavefunctions``: the (up, down) exchange-correlation
    potentials of the last iteration (the core-correction force)."""
    scf_hartree_potential: np.ndarray | None = None
    """With ``keep_wavefunctions``: the Hartree potential of the last
    iteration (the Gaussian-charge term of the wire and slab forces)."""
    hubbard: object | None = None
    """LDA+U: the :class:`parsec_python.Hamiltonian.hubbard.HubbardTerms` of the
    last iteration, ``None`` otherwise (as :class:`SinglePointResult`)."""
    dispersion: object | None = None
    """Van_Der_Waals: the TS correction of the converged density, ``None``
    otherwise (as :class:`SinglePointResult`)."""
    timings: RunTimings | None = None
    """Stage timings, recorded by the k-point driver (``None`` where not measured)."""

    @property
    def magnetic_moment(self) -> float:
        """``N_up - N_down`` in Bohr magnetons (PARSEC's ``<S>`` diagnostic),
        summed with each pooled state's k-point weight."""
        weights = self.k_weights
        if weights is not None and len(weights) > 1 and np.any(
            np.asarray(weights) != np.asarray(weights)[0]
        ):
            # Unequal weights (an irreducible wedge or a weighted list): each
            # k-point's states carry their own weight.
            per_k = np.asarray(self.occupations_up).size // len(weights)
            state_weights = np.repeat(np.asarray(weights, dtype=float), per_k)
            return float(
                np.dot(state_weights, self.occupations_up)
                - np.dot(state_weights, self.occupations_down)
            )
        return float(
            self.occupation_weight
            * (np.sum(self.occupations_up) - np.sum(self.occupations_down))
        )


@dataclass(frozen=True)
class SelfConsistentSOCIteration:
    """Diagnostics for one iteration of
    :func:`~parsec_python.SCF.self_consistent_soc.run_self_consistent_soc`.

    Unlike :class:`SpinPolarizedSCFIteration`, there is only one set of
    (SOC-split, spinor) eigenpairs per iteration -- spin is not a separate
    axis here, it is folded into each ``2*n_grid``-length eigenvector.
    """

    iteration: int
    weighted_residual: float
    plain_residual: float
    energies: EnergyBreakdown
    eigenvalues: tuple[float, ...] = ()
    occupations: tuple[float, ...] = ()
    fermi_level: float = float("nan")
    # Non-collinear runs only: the moments bxc.F90 prints after flevel
    # (parsec_python.SCF.noncollinear.NoncollinearMoments).
    magnetic_moments: object | None = None
    # Symmetrize_Charge_Density of a periodic spin-orbit run: the symm_scalar
    # reports (total density, then up and down where spin-polarized).
    density_symmetrization: tuple = ()
    # LDA+U only, from the second iteration on (Ry), as in SCFIteration.
    hubbard_first_contribution: float | None = None
    hubbard_energy: float | None = None
    # Dynamic_Diag_Tol only: the eigensolver tolerance of the next iteration.
    next_eigensolver_tolerance: float | None = None
    eigen_residual_max: float = float("nan")
    """Largest ``||H x - e x||`` of the requested pairs after this iteration's solve (NaN if not measured)."""
    diagonalization_seconds: float = 0.0
    total_seconds: float = 0.0
    """Seconds of this iteration's diagonalization and of the whole iteration (0 where not measured)."""


@dataclass
class SelfConsistentSOCResult:
    """Result of :func:`~parsec_python.SCF.self_consistent_soc.run_self_consistent_soc`.

    Fortran's ``SO_from_scratch`` without ``Non_Collinear_magnetism``: SOC is
    in the Hamiltonian from the first SCF iteration (unlike
    :class:`PerturbativeSpinOrbitResult`, which only ever diagonalizes a
    small fixed-basis matrix after an ordinary scalar SCF converges), but
    both spinor components still share one spin-unpolarized effective
    potential (no ``B_xc.sigma`` term) -- see
    :mod:`parsec_python.Hamiltonian.spinor_operator`'s module docstring.
    """

    converged: bool
    iterations: int
    atoms: tuple[Atom, ...]
    electron_count: float
    eigenvalues: np.ndarray
    occupations: np.ndarray
    spinors: np.ndarray
    """Shape ``(2*n_grid, n_states)`` complex: rows ``0:n_grid`` are each
    state's spin-up component, rows ``n_grid:2*n_grid`` are spin-down."""
    fermi_level: float
    density: np.ndarray
    energies: EnergyBreakdown
    magnetic_moment: float | None = None
    """Net ``N_up - N_down`` (Bohr magnetons) from the spin densities when
    the run was spin-polarized; for a non-collinear run, the length of
    ``magnetic_moment_vector``.  ``None`` for the spin-unpolarized drivers."""
    magnetic_moment_vector: np.ndarray | None = None
    """Net moment ``(m_x, m_y, m_z)`` in Bohr magnetons (non-collinear runs)."""
    magnetization: np.ndarray | None = None
    """Magnetization density ``m(r)``, shape ``(3, n_grid)``, in Bohr
    magnetons per bohr^3 (non-collinear runs)."""
    scf_potentials: dict[str, np.ndarray] | None = None
    """Converged local potentials of a periodic run (output side of the last
    iteration), the input of a non-self-consistent band calculation."""
    k_points: np.ndarray | None = None
    """Cartesian k-points, shape ``(n_k, 3)`` in bohr^-1, for a k-point-sampled
    run (``None`` otherwise).  Pooled eigenvalue/occupation arrays are
    k-major: entries ``i*n_states:(i+1)*n_states`` belong to ``k_points[i]``."""
    k_weights: np.ndarray | None = None
    """Normalized k-point weights, shape ``(n_k,)``."""
    state_moments: np.ndarray | None = None
    """``<S_z>`` of every pooled state (same k-major layout as ``eigenvalues``)
    for k-point runs, where ``spinors`` only holds the first k-point."""
    scf_input_potential: np.ndarray | None = None
    """Spin-independent input potential of the last iteration (the channel
    average of collinear spin, the scalar part of a non-collinear potential),
    for the self-consistency term of the forces."""
    scf_output_potential: np.ndarray | None = None
    """The output potential of the last iteration, as ``scf_input_potential``."""
    xc_potential: np.ndarray | None = None
    """Spin-independent exchange-correlation potential of the last output
    (the core-correction term of the forces)."""
    k_spinors: tuple[np.ndarray, ...] | None = None
    """With ``keep_wavefunctions`` of a k-point driver: the converged spinors
    of every k-point, in the order of ``k_points``; ``None`` otherwise."""
    hubbard: object | None = None
    """LDA+U: the spinor terms of the last iteration, ``None`` otherwise."""
    parsec_dat_fields: dict | None = None
    """What ``wfnsave.F90`` stores of a spinor run: ``kind`` (``stype(2)``: 2
    spin-orbit, 3 non-collinear, 4 both), ``potentials`` (``vnew(:,1:2)``, the
    last mixed potential of each spin component, Ry) and ``densities``
    (``rho(:,2:3)``); ``None`` for a result made elsewhere."""
    dispersion: object | None = None
    """Van_Der_Waals: the TS correction of the converged density, ``None``
    otherwise (as :class:`SinglePointResult`)."""
    next_potential: object | None = None
    """The last mixed potential (``pot%vnew`` after the last iteration), in
    the form the driver's ``initial_potential`` takes: one array (spin-orbit),
    the up and down potentials (spin-orbit with collinear spin), or the
    scalar potential and the exchange-correlation field ``(V, B)`` with ``B``
    of shape ``(3, n_grid)`` (non-collinear).  What the next structure of an
    ionic loop starts from (``parsec.F90``: ``vnew - vion_old + vion_new``);
    ``None`` for a result made elsewhere."""
    eigensolver_state: object | None = None
    """The eigensolver subspace of the last iteration of the isolated
    spin-orbit loops (``run_self_consistent_soc`` and its spin-polarized
    form), which the next structure of an ionic loop filters again
    (``solver%firstfilt``); ``None`` for the drivers that keep none."""
    timings: dict[str, float] | None = None
    """Seconds per SCF stage (``hamiltonian``, ``diagonalization``, ``density``,
    ``hartree``, ``xc``, ``mixing_energy``) plus the eigensolver's own counters
    (``eig_*``), for the single-point driver; ``None`` where not recorded."""
    xc_potential: np.ndarray | None = None
    """Spin-averaged exchange-correlation potential (``V_avg``) made from the last density plus the frozen core density:
    what the core-correction force integrates.  ``None`` where a driver does not record it."""
    input_effective_potential: np.ndarray | None = None
    """Spin-averaged local potential that entered the last eigenproblem (``V_in`` of the force term for the SCF residual)."""
    output_effective_potential: np.ndarray | None = None
    """Spin-averaged local potential made from the last density (``V_out``)."""
    nonlocal_forces: np.ndarray | None = None
    """``(atoms, 3)`` nonlocal force (projectors and spin-orbit term) of a k-point run, which keeps no orbitals but the
    first k-point's: accumulated from every k-point's orbitals at the end of the SCF loop (``compute_forces=True``)."""
    next_effective_potential: np.ndarray | tuple | None = None
    """The mixed potential the loop would have used for one more iteration (what the next structure of a relaxation
    or dynamics run starts from, see :func:`parsec_python.SCF.single_point.warm_start_for_moved_atoms`): one array, the
    pair ``(V_up, V_down)`` of a collinear-spin loop, or ``(V, B_xc)`` of a non-collinear one."""
    hartree_potential: np.ndarray | None = None
    """Hartree potential of the last density: the starting vector of the first Poisson solve of the next structure."""
    eigensolver_state: object | None = None
    """The eigensubspace the next structure continues from: one state for a cluster, a ``{k-point: state}`` dict for a
    k-point run (held on the host)."""

    @property
    def magnetic_moment_per_state(self) -> np.ndarray:
        """Each SOC-split state's own ``<S_z>`` (up-component weight minus
        down-component weight), matching ``elec_st%magmom``.  For a
        k-point run this covers every pooled state (``state_moments``)."""
        if self.state_moments is not None:
            return self.state_moments
        n_grid = self.spinors.shape[0] // 2
        up_weight = np.sum(np.abs(self.spinors[:n_grid, :]) ** 2, axis=0)
        down_weight = np.sum(np.abs(self.spinors[n_grid:, :]) ** 2, axis=0)
        return up_weight - down_weight
