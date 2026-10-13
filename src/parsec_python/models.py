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


DomainShape = Literal["sphere", "box"]
EigensolverName = Literal["chebff", "chebdav", "arpack"]
HartreeBoundaryMethod = Literal["auto", "multipole", "direct"]
HartreeAtomicTail = Literal["auto", "on", "off"]
# The largest multipole order of the Hartree boundary every builder accepts.
# The recurrences are sound beyond it; the normalization prefactor of Y_lm
# becomes subnormal at l = m = 86 and would silently drop terms.
MAXIMUM_MULTIPOLE_ORDER = 60
XCFunctional = Literal["ca", "pbe"]
InitialDensityMethod = Literal["sad", "file", "charge3net", "scdp"]
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

    def __post_init__(self) -> None:
        symbol = self.symbol.strip()
        position = np.asarray(self.position, dtype=float)
        if not symbol:
            raise ValueError("atom symbol cannot be empty")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("atom position must contain three finite Cartesian coordinates")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "position", position)


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
    """

    path: str | Path
    local_angular_momentum: int
    read_valence_density: bool = False
    use_spline: bool = False
    element_symbol: str | None = None
    atomic_energy_correction: float = 0.0
    pseudopotential_format: str = "martins_new"
    number_of_channels: int | None = None
    electrons_per_orbital: tuple[float, ...] | None = None
    core_cutoff_radius: float | None = None
    initial_spin_polarization: float = DEFAULT_INITIAL_SPIN_POLARIZATION

    def __repr__(self) -> str:
        # A species that leaves the starting polarization at PARSEC's default
        # prints what it printed before the field existed.
        shown = [
            f"{item.name}={getattr(self, item.name)!r}"
            for item in fields(self)
            if item.name != "initial_spin_polarization"
            or self.initial_spin_polarization != DEFAULT_INITIAL_SPIN_POLARIZATION
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
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "local_angular_momentum", local_l)
        object.__setattr__(
            self,
            "element_symbol",
            None if element is None else element.strip(),
        )
        object.__setattr__(self, "atomic_energy_correction", correction)

    def _check_format_input(self, local_l: int) -> None:
        """Validate what the input adds to a pseudopotential file."""

        pseudo_format = str(self.pseudopotential_format).strip().lower()
        if pseudo_format not in {"martins_new", "martins_old"}:
            raise ValueError(
                "pseudopotential_format must be 'martins_new' or 'martins_old'"
            )
        object.__setattr__(self, "pseudopotential_format", pseudo_format)
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
        if any(value is None for value in given):
            raise ValueError(
                "a martins_old pseudopotential needs number_of_channels, "
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


@dataclass(frozen=True)
class GridSettings:
    """PARSEC cluster-grid settings.

    ``expansion_order`` is the user-facing PARSEC order.  It must be even;
    half that many neighbors are used on each side of a point.
    """

    spacing: float
    radius: float
    expansion_order: int = 12
    shift: tuple[float, float, float] = (0.5, 0.5, 0.5)
    domain_shape: DomainShape = "sphere"
    box_lengths: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("grid spacing must be positive")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("domain radius must be positive")
        order = int(self.expansion_order)
        if order != self.expansion_order:
            raise ValueError("PARSEC expansion_order must be an integer")
        if order < 2 or order > 20 or order % 2:
            raise ValueError("PARSEC expansion_order must be an even integer from 2 to 20")
        if len(self.shift) != 3 or not np.all(np.isfinite(self.shift)):
            raise ValueError("grid shift must have three finite entries")
        if self.domain_shape not in {"sphere", "box"}:
            raise ValueError("only sphere and box cluster domains are supported")
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

    @property
    def stencil_half_width(self) -> int:
        return self.expansion_order // 2

    @property
    def enclosing_radius(self) -> float:
        """PARSEC ``rmax`` used only to size the underlying Cartesian box."""
        if self.domain_shape == "box":
            return 0.5 * float(np.linalg.norm(np.asarray(self.box_lengths)))
        return self.radius


@dataclass(frozen=True)
class PeriodicCell:
    """Real-space lattice vectors for a periodic unit cell, in bohr.
    """

    lattice_vectors: np.ndarray
    """Rows are the lattice vectors ``a1``, ``a2``, ``a3``."""

    def __post_init__(self) -> None:
        vectors = np.asarray(self.lattice_vectors, dtype=np.float64)
        if vectors.shape != (3, 3):
            raise ValueError("lattice_vectors must have shape (3, 3)")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("lattice_vectors must be finite")
        if abs(float(np.linalg.det(vectors))) <= 0.0:
            raise ValueError("lattice_vectors must span a nondegenerate cell")
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
    domain_shape: DomainShape = "PBC_Orthorhombic"
    box_lengths: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("grid spacing must be positive")
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
    CHEBDAV.  ``first_filter_cycles`` is ``FF_MaxIter`` and affects CHEBFF
    only; CHEBDAV instead uses ``tolerance`` for residual locking.
    ``filter_degree`` and ``filter_degree_delta`` control the later subspace
    filter.  The input translator resolves PARSEC's ``Matvec_Blocksize``
    default from ``States_Num`` and stores the resulting positive integer in
    ``matvec_block_size``.
    """

    method: EigensolverName = "chebff"
    tolerance: float = 1.0e-4
    first_filter_degree: int = 20
    first_filter_cycles: int = 2
    matvec_block_size: int = 6
    subspace_buffer: int = 6
    filter_degree: int = 15
    filter_degree_delta: int = 3
    lanczos_steps: int = 5
    random_seed: int = 7

    def __post_init__(self) -> None:
        if self.method not in {"chebff", "chebdav", "arpack"}:
            raise ValueError(
                "eigensolver method must be 'chebff', 'chebdav', or 'arpack'"
            )
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
        if self.first_filter_degree < minimum_first_degree:
            raise ValueError(
                f"first_filter_degree must be at least {minimum_first_degree} "
                f"for {self.method}"
            )
        if not 1 <= self.first_filter_cycles <= 9:
            raise ValueError("first_filter_cycles must be between 1 and 9")
        if self.matvec_block_size < 1:
            raise ValueError("matvec_block_size must be positive")
        if self.subspace_buffer < 0:
            raise ValueError("subspace_buffer cannot be negative")
        if self.method in {"chebff", "chebdav"} and self.subspace_buffer < 6:
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


@dataclass(frozen=True)
class MixingSettings:
    """Anderson mixing settings used by the SCF driver."""

    parameter: float = 0.30
    memory: int = 4
    restart: int = 20
    regularization: float = 0.0
    safeguard: bool = False
    step_limit: float = 2.0
    growth_trigger: float = 2.0
    backoff: float = 0.5

    def __post_init__(self) -> None:
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


@dataclass(frozen=True)
class SCFSettings:
    """Isolated single-point SCF controls.

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

    def __repr__(self) -> str:
        # The text of these settings is hashed by the cache keys of the
        # accelerated driver and stored in test records.  Occupations that
        # are not set and spin polarization that is off print nothing, so
        # that the text of every calculation without them stays what it was
        # before the fields existed.
        hidden = {
            "fixed_occupations": self.fixed_occupations is None,
            "spin_polarized": not self.spin_polarized,
            "fixed_occupations_down": self.fixed_occupations_down is None,
        }
        shown = [
            f"{item.name}={getattr(self, item.name)!r}"
            for item in fields(self)
            if not hidden.get(item.name, False)
        ]
        return f"{type(self).__qualname__}({', '.join(shown)})"

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
        if self.xc_functional not in {"ca", "pbe"}:
            raise ValueError("xc_functional must be 'ca' or 'pbe'")
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
        object.__setattr__(self, "max_iterations", max_iterations)


@dataclass(frozen=True)
class InitialDensitySettings:
    """Select and configure the valence-density SCF initial guess.

    ``sad`` is the PARSEC-compatible superposition of atomic densities and is
    the unchanged default.  ``file`` imports a portable ``.npz`` field or a
    legacy three-dimensional ``.npy`` field.  ``charge3net`` and ``scdp`` run
    an isolated model environment through the optional :mod:`MLDensity`
    adapters and cache their prediction as the same portable ``.npz`` format.
    Supplying ``file`` with either named provider imports an earlier prediction
    from that provider without launching its environment.

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

    def __post_init__(self) -> None:
        method = str(self.method).strip().lower()
        if method == "ml":
            method = "file"
        if method not in {"sad", "file", "charge3net", "scdp"}:
            raise ValueError(
                "initial density method must be sad, file, charge3net, or scdp"
            )
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
        if method == "charge3net" and model not in {"qm9", "mp", "nmc"}:
            raise ValueError("ChargE3Net model must be qm9, mp, or nmc")
        if method == "scdp" and model not in {
            "qm9",
            "fast",
            "accurate",
            "qm9-fast",
            "qm9-accurate",
        }:
            raise ValueError("SCDP model must be fast or accurate")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "prediction_chunk_size", chunk_size)
        for name in (
            "file",
            "repository",
            "checkpoint",
            "python_executable",
            "cache_directory",
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
    ``Relax_Restart``.
    """

    method: RelaxationMethod
    max_movements: int = 500
    minimum_force: float = 1.0e-2
    maximum_step: float = DEFAULT_RELAXATION_MAXIMUM_STEP
    minimum_step: float = 1.0e-4
    bfgs_corrections: int = 7
    restart: bool = False

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
    # Buffered eigensubspace of the last eigenproblem.  An ionic step hands
    # it to the SCF loop of the next structure; nothing else reads it.
    eigensolver_state: object | None = None
    # The two channels of a spin-polarized calculation, up first.
    spin: tuple[SpinChannelResult, SpinChannelResult] | None = None

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
