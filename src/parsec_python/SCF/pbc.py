"""
Periodic (Gamma-point) SCF preparation, reusing the isolated SCF loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np
import scipy.sparse as sp

from ..Grid.pbc import PeriodicRealSpaceGrid, build_periodic_grid
from ..Hamiltonian import KohnShamHamiltonian
from ..Hartree.pbc import PeriodicHartreeResult, solve_periodic_hartree
from ..Laplacian import build_negative_laplacian
from ..models import (
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
from ..V_xc import XCResult, ca_lda
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
    ionic_potential: np.ndarray
    nonlocal_operator: NonlocalProjectorOperator
    initial_density: np.ndarray
    core_density: np.ndarray
    ion_ion_energy: float
    atomic_reference_correction: float = 0.0
    alpha_z_energy: float = 0.0
    timings: PreparationTimings = field(default_factory=PreparationTimings)

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
        raise NotImplementedError(
            f"xc_functional={self.input.scf.xc_functional!r} is not supported for "
            "the periodic path: pbe's density gradient zero-pads outside the "
            "active domain (the isolated convention), not periodic wraparound"
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
    side_lengths = np.diag(problem.periodic_cell.lattice_vectors)
    for atom in atoms:
        position = np.asarray(atom.position, dtype=float)
        if np.any(position < 0.0) or np.any(position >= side_lengths):
            raise ValueError(
                f"atom {atom.symbol} at {position} lies outside the periodic "
                f"cell [0, {side_lengths.tolist()}); this path does not "
                "recenter geometry the way the isolated path does"
            )

    stage_start = time.perf_counter()
    negative_laplacian = build_negative_laplacian(grid)
    finite_difference_seconds = time.perf_counter() - stage_start

    # 3. Local and Kleinman--Bylander nonlocal ionic terms.
    stage_start = time.perf_counter()
    ionic_potential = ewald_local_ionic_potential(
        grid,
        atoms,
        pseudopotentials,
        problem.pseudopotentials,
        problem.periodic_cell.lattice_vectors,
    )
    local_ionic_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    nonlocal_operator = build_nonlocal_projectors(
        grid,
        atoms,
        pseudopotentials,
        problem.pseudopotentials,
        problem.periodic_cell.lattice_vectors,
    )
    nonlocal_ionic_seconds = time.perf_counter() - stage_start

    # 4. Initial valence density and nonlinear core-correction density.  SAD
    # remains the PARSEC-compatible default. A file/ML provider is not implemented yet
    stage_start = time.perf_counter()
    if problem.initial_density_settings.method != "sad":
        raise NotImplementedError(
            "the periodic path only supports "
            "initial_density_settings.method='sad'"
        )
    initial_density = superpose_atomic_density(
        grid,
        atoms,
        pseudopotentials,
        problem.pseudopotentials,
        problem.periodic_cell.lattice_vectors,
    )
    if problem.scf.normalize_initial_density:
        initial_density = normalize_density(initial_density, grid, electron_count)
    initial_density_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    core_density = superpose_atomic_density(
        grid,
        atoms,
        pseudopotentials,
        problem.pseudopotentials,
        problem.periodic_cell.lattice_vectors,
        core=True,
    )
    core_density_seconds = time.perf_counter() - stage_start

    # 5. Geometry-only ion--ion contribution to the total energy.
    stage_start = time.perf_counter()
    repulsion = ewald_ion_ion_energy(
        atoms, pseudopotentials, problem.periodic_cell.lattice_vectors
    )
    alpha_z_energy = ewald_alpha_z_energy(
        atoms,
        pseudopotentials,
        problem.periodic_cell.lattice_vectors,
        electron_count,
    )
    atomic_reference_correction = float(
        sum(
            problem.pseudopotentials[atom.symbol].atomic_energy_correction
            for atom in atoms
        )
    )
    ion_ion_seconds = time.perf_counter() - stage_start

    preparation_timings = PreparationTimings(
        pseudopotential_loading_seconds=pseudopotential_loading_seconds,
        grid_seconds=grid_seconds,
        finite_difference_seconds=finite_difference_seconds,
        local_ionic_seconds=local_ionic_seconds,
        nonlocal_ionic_seconds=nonlocal_ionic_seconds,
        initial_density_seconds=initial_density_seconds,
        core_density_seconds=core_density_seconds,
        ion_ion_seconds=ion_ion_seconds,
        total_seconds=time.perf_counter() - preparation_start,
    )
    return PeriodicPreparedSinglePointSystem(
        input=problem,
        atoms=atoms,
        electron_count=electron_count,
        pseudopotentials=pseudopotentials,
        grid=grid,
        negative_laplacian=negative_laplacian,
        ionic_potential=ionic_potential,
        nonlocal_operator=nonlocal_operator,
        initial_density=initial_density,
        core_density=core_density,
        ion_ion_energy=repulsion,
        atomic_reference_correction=atomic_reference_correction,
        alpha_z_energy=alpha_z_energy,
        timings=preparation_timings,
    )


__all__ = [
    "PeriodicPreparedSinglePointSystem",
    "prepare_periodic_single_point",
]
