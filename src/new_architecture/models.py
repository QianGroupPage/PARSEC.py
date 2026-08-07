"""Public data models for the modular PARSEC-style single-point solver.

All lengths are in bohr, densities are in electrons/bohr**3, and energies and
potentials are in Rydberg unless a field explicitly says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np


DomainShape = Literal["sphere", "box"]
EigensolverName = Literal["chebff", "chebdav", "arpack"]
HartreeBoundaryMethod = Literal["auto", "multipole", "direct"]


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
    """Location and input-only choices for one PARSEC pseudopotential."""

    path: str | Path
    local_angular_momentum: int
    read_valence_density: bool = False
    use_spline: bool = False

    def __post_init__(self) -> None:
        local_l = int(self.local_angular_momentum)
        if local_l != self.local_angular_momentum or not 0 <= local_l <= 3:
            raise ValueError("local angular momentum must be 0 (s) through 3 (f)")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "local_angular_momentum", local_l)


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
class HartreeSettings:
    """Finite-cluster Poisson settings matching ``hartset``/``hpotcg``."""

    multipole_order: int = 9
    relative_tolerance: float = 1.0e-7
    absolute_tolerance: float = 1.0e-13
    max_iterations: int = 1600
    boundary_method: HartreeBoundaryMethod = "auto"
    direct_chunk_size: int = 16

    def __post_init__(self) -> None:
        order = int(self.multipole_order)
        max_iterations = int(self.max_iterations)
        direct_chunk_size = int(self.direct_chunk_size)
        if order != self.multipole_order or not 0 <= order <= 9:
            raise ValueError("multipole_order must be between 0 and 9")
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
        object.__setattr__(self, "memory", memory)
        object.__setattr__(self, "restart", restart)


@dataclass(frozen=True)
class SCFSettings:
    """Spin-unpolarized, isolated single-point SCF controls."""

    max_iterations: int = 50
    convergence_criterion: float = 2.0e-4
    fermi_temperature_kelvin: float = 80.0
    number_of_states: int | None = None
    net_charge: float = 0.0
    use_plain_residual: bool = False
    normalize_initial_density: bool = True

    def __post_init__(self) -> None:
        max_iterations = int(self.max_iterations)
        if max_iterations != self.max_iterations or max_iterations < 1:
            raise ValueError("SCF max_iterations must be positive")
        if (
            not np.isfinite(self.convergence_criterion)
            or self.convergence_criterion <= 0
        ):
            raise ValueError("SCF convergence criterion must be positive")
        if (
            not np.isfinite(self.fermi_temperature_kelvin)
            or self.fermi_temperature_kelvin < 0
        ):
            raise ValueError("negative Fermi temperature/file occupations are not supported")
        if not np.isfinite(self.net_charge):
            raise ValueError("net_charge must be finite")
        if self.number_of_states is not None:
            states = int(self.number_of_states)
            if states != self.number_of_states or states < 1:
                raise ValueError("number_of_states must be a positive integer")
            object.__setattr__(self, "number_of_states", states)
        object.__setattr__(self, "max_iterations", max_iterations)


@dataclass(frozen=True)
class SinglePointInput:
    """Complete input to the modular isolated single-point calculator."""

    atoms: Sequence[Atom]
    pseudopotentials: Mapping[str, SpeciesPotential]
    grid: GridSettings
    scf: SCFSettings = field(default_factory=SCFSettings)
    hartree: HartreeSettings = field(default_factory=HartreeSettings)
    eigensolver: EigensolverSettings = field(default_factory=EigensolverSettings)
    mixing: MixingSettings = field(default_factory=MixingSettings)
    recenter_geometry: bool = True

    def __post_init__(self) -> None:
        atoms = tuple(atom if isinstance(atom, Atom) else Atom(**atom) for atom in self.atoms)
        if not atoms:
            raise ValueError("a single-point calculation requires at least one atom")
        missing = sorted({atom.symbol for atom in atoms}.difference(self.pseudopotentials))
        if missing:
            raise ValueError(f"missing pseudopotential specifications for: {', '.join(missing)}")
        object.__setattr__(self, "atoms", atoms)


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
    fermi_level: float = float("nan")
    density_minimum: float = float("nan")
    density_maximum: float = float("nan")
    diagonalization_seconds: float = 0.0
    hartree_seconds: float = 0.0


@dataclass
class SinglePointResult:
    """Result of a modular isolated single-point calculation."""

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
