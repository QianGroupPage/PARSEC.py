"""
Periodic (Gamma-point) SCF preparation, reusing the isolated SCF loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Callable

import numpy as np
import scipy.sparse as sp

from ..Grid.pbc import PeriodicRealSpaceGrid, build_periodic_grid
from ..Hamiltonian import KohnShamHamiltonian
from ..Hartree.pbc import PeriodicHartreeResult, solve_periodic_hartree
from ..Laplacian import build_gradient, build_negative_laplacian
from ..models import (
    Atom,
    PreparationTimings,
    SCFIteration,
    SinglePointInput,
    SinglePointResult,
)
from ..Pseudopotential import ParsecPseudopotential
from ..V_ion import (
    NonlocalProjectorOperator,
    build_nonlocal_projectors,
    ewald_alpha_z_energy,
    ewald_ion_ion_energy,
    ewald_local_ionic_potential,
    ionic_charge,
    load_pseudopotentials,
    normalize_density,
    superpose_atomic_density,
)
from ..V_xc import (
    NoncollinearXCResult,
    SpinPolarizedXCResult,
    SpinXCResult,
    XCResult,
    ca_lda,
    ca_lda_spin,
    ca_lda_spin_polarized,
    pbe,
    noncollinear_xc,
    pbe_spin,
    pbe_spin_polarized,
)
from .single_point import _species_densities, initial_spin_densities
from .single_point import run_scf as _run_scf


@dataclass(frozen=True)
class PeriodicPreparedSinglePointSystem:
    """Made to mirror of PreparedSinglePointSystem in single_point.py, should eitheir be made from a base class or folded into single_point.py
    """

    input: SinglePointInput
    atoms: tuple
    electron_count: float
    pseudopotentials: dict[str, ParsecPseudopotential]
    grid: PeriodicRealSpaceGrid
    negative_laplacian: sp.csr_matrix
    gradient: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]
    ionic_potential: np.ndarray
    nonlocal_operator: NonlocalProjectorOperator
    initial_density: np.ndarray
    core_density: np.ndarray
    ion_ion_energy: float
    atomic_reference_correction: float = 0.0
    alpha_z_energy: float = 0.0
    timings: PreparationTimings = field(default_factory=PreparationTimings)
    initial_spin_densities: tuple[np.ndarray, np.ndarray] | None = None
    """Starting densities of the up and the down channel of a ``Spin_Polarization`` calculation (``initchrg.f90``)."""

    @property
    def cell_atoms(self) -> tuple:
        """The atoms folded into the cell, in the order of ``atoms`` (see :func:`fold_atoms_into_cell`).

        ``atoms`` keeps the coordinates a relaxation or a dynamics run moves them to, which leave the cell; every
        builder of an image sum (local potential, projectors, densities, Ewald sums, forces) assumes atoms inside
        it and gets these.
        """
        return fold_atoms_into_cell(self.atoms, self.input.periodic_cell.lattice_vectors)

    def hamiltonian(self, effective_potential: np.ndarray) -> KohnShamHamiltonian:
        """Compose the current SCF Hamiltonian without forming a dense matrix.

        The supplied array is only the diagonal local field

        ``V_eff = V_ion,local + V_H + V_xc``.

        :class:`KohnShamHamiltonian` binds it to the already prepared kinetic
        and nonlocal terms, giving

        ``H = -nabla_FD^2 + diag(V_eff) + V_NL``.
        """
        return KohnShamHamiltonian(
            self.negative_laplacian,
            effective_potential,
            self.nonlocal_operator,
        )

    def solve_hartree(
        self,
        density: np.ndarray,
        initial_potential: np.ndarray | None = None,
        *,
        raise_on_nonconvergence: bool = True,
    ) -> PeriodicHartreeResult:
        return solve_periodic_hartree(
            density,
            self.grid,
            self.negative_laplacian,
            self.input.hartree,
            initial_potential,
            raise_on_nonconvergence=raise_on_nonconvergence,
        )

    def evaluate_xc(self, density: np.ndarray) -> XCResult:
        """Evaluate the selected XC functional including frozen NLCC."""

        if self.input.scf.xc_functional == "ca":
            return ca_lda(density, self.grid.volume_element, self.core_density)
        if self.input.scf.xc_functional == "pbe":
            return pbe(density, self.grid, self.core_density)
        raise NotImplementedError(
            f"xc_functional={self.input.scf.xc_functional!r} is not supported for "
            "the periodic path"
        )

    def evaluate_spin_xc(self, density_up: np.ndarray, density_down: np.ndarray) -> SpinXCResult:
        """The spin-polarized XC functional of the loop of :func:`parsec_python.SCF.spin_polarized.run_spin_scf`."""

        functional = self.input.scf.xc_functional
        if functional == "ca":
            return ca_lda_spin(density_up, density_down, self.grid.volume_element, self.core_density)
        if functional == "pbe":
            return pbe_spin(density_up, density_down, self.grid, self.core_density)
        raise ValueError(f"unsupported XC functional {functional!r}")

    def evaluate_xc_spin_polarized(
        self, density_up: np.ndarray, density_down: np.ndarray
    ) -> SpinPolarizedXCResult:
        """Collinear spin-polarized XC (LSDA or PBE) including frozen NLCC."""

        functional = self.input.scf.xc_functional
        if functional == "ca":
            return ca_lda_spin_polarized(
                density_up,
                density_down,
                self.grid.volume_element,
                self.core_density,
            )
        if functional == "pbe":
            return pbe_spin_polarized(
                density_up, density_down, self.grid, self.core_density
            )
        raise ValueError(f"unsupported XC functional {functional!r}")

    def evaluate_xc_noncollinear(
        self, density: np.ndarray, magnetization: np.ndarray
    ) -> NoncollinearXCResult:
        """Non-collinear XC (``V_avg`` and ``B_xc``) from ``n`` and ``m(r)``."""

        return noncollinear_xc(density, magnetization, self.evaluate_xc_spin_polarized)


def fold_atoms_into_cell(atoms, lattice_vectors: np.ndarray) -> tuple:
    """Atoms with their positions moved by lattice vectors into the cell (fractional coordinates in ``[0, 1)``).

    A periodic image of an atom is the same atom: the energy and the forces do not depend on which one the
    coordinates name, but the builders of image sums are written for atoms inside the cell.  Atoms already inside are
    returned as they are, bit for bit.
    """

    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    inverse = np.linalg.inv(lattice)
    folded = []
    for atom in atoms:
        position = np.asarray(atom.position, dtype=np.float64)
        fraction = position @ inverse
        inside = np.all(fraction >= -1.0e-12) and np.all(fraction < 1.0 + 1.0e-12)
        if inside:
            folded.append(atom)
            continue
        wrapped = fraction - np.floor(fraction)
        folded.append(replace(atom, position=wrapped @ lattice))
    return tuple(folded)


def _geometry_terms(problem: SinglePointInput, grid, atoms, pseudopotentials, electron_count: float) -> dict:
    """Everything of a periodic system that depends on the atom positions, for atoms inside the cell.

    The local ionic potential, the projectors with their images, the atomic starting density and the core density, the
    Ewald ion-ion energy and the alpha-Z term, with the seconds of each stage (``PreparationTimings`` fields).
    """

    lattice_vectors = problem.periodic_cell.lattice_vectors
    stage_start = time.perf_counter()
    ionic_potential = ewald_local_ionic_potential(
        grid, atoms, pseudopotentials, problem.pseudopotentials, lattice_vectors
    )
    local_ionic_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    nonlocal_operator = build_nonlocal_projectors(
        grid, atoms, pseudopotentials, problem.pseudopotentials, lattice_vectors
    )
    nonlocal_ionic_seconds = time.perf_counter() - stage_start

    # Initial valence density and nonlinear core-correction density.  SAD
    # remains the PARSEC-compatible default. A file/ML provider is not implemented yet
    stage_start = time.perf_counter()
    if problem.initial_density_settings.method != "sad":
        raise NotImplementedError(
            "the periodic path only supports "
            "initial_density_settings.method='sad'"
        )
    initial_density = superpose_atomic_density(
        grid, atoms, pseudopotentials, problem.pseudopotentials, lattice_vectors
    )
    spin_densities = None
    if problem.scf.spin_polarized:
        # initchrg.f90: the two channels carry the factor that normalizes the total density (see initial_spin_densities)
        def superpose(grid_, atoms_, pseudopotentials_, specifications_):
            return superpose_atomic_density(grid_, atoms_, pseudopotentials_, specifications_, lattice_vectors)

        spin_densities = initial_spin_densities(
            initial_density,
            _species_densities(initial_density, superpose, grid, atoms, pseudopotentials, problem.pseudopotentials),
            {symbol: specification.initial_spin_polarization for symbol, specification in problem.pseudopotentials.items()},
            electron_count / grid.integrate(initial_density) if problem.scf.normalize_initial_density else 1.0,
        )
    if problem.scf.normalize_initial_density:
        initial_density = normalize_density(initial_density, grid, electron_count)
    initial_density_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    core_density = superpose_atomic_density(
        grid, atoms, pseudopotentials, problem.pseudopotentials, lattice_vectors, core=True
    )
    core_density_seconds = time.perf_counter() - stage_start

    # Geometry-only ion--ion contribution to the total energy.
    stage_start = time.perf_counter()
    repulsion = ewald_ion_ion_energy(atoms, pseudopotentials, lattice_vectors)
    alpha_z_energy = ewald_alpha_z_energy(atoms, pseudopotentials, lattice_vectors, electron_count)
    ion_ion_seconds = time.perf_counter() - stage_start
    return {
        "fields": dict(
            ionic_potential=ionic_potential,
            nonlocal_operator=nonlocal_operator,
            initial_density=initial_density,
            core_density=core_density,
            ion_ion_energy=repulsion,
            alpha_z_energy=alpha_z_energy,
            initial_spin_densities=spin_densities,
        ),
        "seconds": dict(
            local_ionic_seconds=local_ionic_seconds,
            nonlocal_ionic_seconds=nonlocal_ionic_seconds,
            initial_density_seconds=initial_density_seconds,
            core_density_seconds=core_density_seconds,
            ion_ion_seconds=ion_ion_seconds,
        ),
    }


def update_periodic_geometry(system: PeriodicPreparedSinglePointSystem, atoms) -> PeriodicPreparedSinglePointSystem:
    """``system`` with its atoms at new positions, on the same grid (what a relaxation or dynamics step rebuilds).

    The local ionic potential, the projectors, the starting and core densities and the ion-ion energy are made for
    the new positions; the grid, the kinetic operator, the pseudopotentials and the electron count are shared.  The
    positions are kept as given, also outside the cell (so that an optimizer or an integrator sees continuous
    coordinates); the builders get them folded into the cell (:attr:`PeriodicPreparedSinglePointSystem.cell_atoms`).
    """

    started = time.perf_counter()
    atoms = tuple(atoms)
    if len(atoms) != len(system.atoms) or any(new.symbol != old.symbol for new, old in zip(atoms, system.atoms)):
        raise ValueError("moved atoms must keep their number, order and species")
    problem = replace(system.input, atoms=atoms)
    folded = fold_atoms_into_cell(atoms, problem.periodic_cell.lattice_vectors)
    terms = _geometry_terms(problem, system.grid, folded, system.pseudopotentials, system.electron_count)
    return replace(
        system,
        input=problem,
        atoms=atoms,
        **terms["fields"],
        timings=PreparationTimings(**terms["seconds"], total_seconds=time.perf_counter() - started),
    )


def prepare_periodic_single_point(
    problem: SinglePointInput,
) -> PeriodicPreparedSinglePointSystem:
    """Build every static periodic component, but do not enter the SCF loop.

    ``problem.periodic_cell`` must be set and ``problem.grid`` must be a
    :class:`~..models.PeriodicGridSettings` (enforced by
    ``SinglePointInput.__post_init__``).  Atom positions must already be
    expressed in the periodic grid's own coordinate convention -- Cartesian
    bohr with each component in ``[0, side_length)`` -- since this path does
    not recenter geometry the way the isolated path does (that convention
    is specific to a sphere/box centered on the origin, and would be wrong
    here).
    """
    if problem.periodic_cell is None:
        raise ValueError("prepare_periodic_single_point requires problem.periodic_cell")
    preparation_start = time.perf_counter()
    # 1. Geometry, pseudopotentials, and electron count.
    atoms = tuple(problem.atoms)

    stage_start = time.perf_counter()
    pseudopotentials = load_pseudopotentials(
        problem.pseudopotentials,
        xc_functional=problem.scf.xc_functional
    )
    pseudopotential_loading_seconds = time.perf_counter() - stage_start
    electron_count = ionic_charge(atoms, pseudopotentials) - problem.scf.net_charge
    if electron_count <= 0:
        raise ValueError("the requested system has no valence electrons")

    # 2. Real-space domain and finite-difference kinetic operator.  In
    # Rydberg units this full sparse operator is T=-nabla_FD^2.  It is static
    # and reused in every eigensolver Hamiltonian application.
    stage_start = time.perf_counter()
    grid = build_periodic_grid(problem.periodic_cell, problem.grid)
    grid_seconds = time.perf_counter() - stage_start
    lattice_vectors = problem.periodic_cell.lattice_vectors
    orthorhombic = np.allclose(lattice_vectors, np.diag(np.diag(lattice_vectors)), atol=1.0e-10)
    side_lengths = np.diag(lattice_vectors)
    inverse_lattice = None if orthorhombic else np.linalg.inv(lattice_vectors)
    for atom in atoms:
        position = np.asarray(atom.position, dtype=float)
        if orthorhombic:
            outside = np.any(position < 0.0) or np.any(position >= side_lengths)
        else:
            fraction = position @ inverse_lattice
            outside = np.any(fraction < -1.0e-12) or np.any(fraction >= 1.0 + 1.0e-12)
        if outside:
            where = (
                f"cell [0, {side_lengths.tolist()})"
                if orthorhombic
                else "the cell (fractional coordinates in [0, 1))"
            )
            raise ValueError(
                f"atom {atom.symbol} at {position} lies outside the periodic "
                f"{where}; this path does not "
                "recenter geometry the way the isolated path does"
            )

    stage_start = time.perf_counter()
    negative_laplacian = build_negative_laplacian(grid)
    # Only needed for k-point sampling's -2i*k.grad kinetic term
    # (Hamiltonian.kpoint_operator); computed unconditionally here since it
    # is cheap relative to the Laplacian and keeps this prepared system
    # self-contained for either the Gamma-only or k-point SCF path.
    gradient = build_gradient(grid)
    finite_difference_seconds = time.perf_counter() - stage_start

    # 3. The terms that depend on the positions: local and Kleinman--Bylander nonlocal ionic terms, the starting and
    # core densities, the ion--ion energy.
    terms = _geometry_terms(problem, grid, atoms, pseudopotentials, electron_count)
    atomic_reference_correction = float(
        sum(
            problem.pseudopotentials[atom.symbol].atomic_energy_correction
            for atom in atoms
        )
    )

    preparation_timings = PreparationTimings(
        pseudopotential_loading_seconds=pseudopotential_loading_seconds,
        grid_seconds=grid_seconds,
        finite_difference_seconds=finite_difference_seconds,
        **terms["seconds"],
        total_seconds=time.perf_counter() - preparation_start,
    )
    return PeriodicPreparedSinglePointSystem(
        input=problem,
        atoms=atoms,
        electron_count=electron_count,
        pseudopotentials=pseudopotentials,
        grid=grid,
        negative_laplacian=negative_laplacian,
        gradient=gradient,
        **terms["fields"],
        atomic_reference_correction=atomic_reference_correction,
        timings=preparation_timings,
    )


__all__ = [
    "PeriodicPreparedSinglePointSystem",
    "fold_atoms_into_cell",
    "prepare_periodic_single_point",
    "update_periodic_geometry",
]
