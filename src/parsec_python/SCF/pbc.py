"""
Periodic (Gamma-point) SCF preparation, reusing the isolated SCF loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import threading
import time
from typing import Callable

import numpy as np
import scipy.sparse as sp

from ..Grid.cell import wrap_into_cell
from ..Grid.pbc import PeriodicRealSpaceGrid, build_periodic_grid, uses_fortran_routines
from ..Grid.point_group import GridPointGroup
from ..Grid.space_group import identity_space_group, periodic_grid_symmetry
from ..Hamiltonian import KohnShamHamiltonian
from ..Hamiltonian.hubbard import HubbardProjectors, build_hubbard_projectors
from ..Hartree.pbc import PeriodicHartreeResult, solve_periodic_hartree
from ..Hartree.slab import solve_slab_hartree
from ..Laplacian import build_gradient, build_negative_laplacian
from ..Laplacian.nonorthogonal import (
    NonorthogonalLaplacianCoefficients,
    grid_laplacian_coefficients,
)
from ..models import (
    Atom,
    PreparationTimings,
    SCFIteration,
    SinglePointInput,
    SinglePointResult,
)
from ..Pseudopotential import ParsecPseudopotential
from ..V_ion.slab import slab_ewald_energy, slab_local_ionic_potential
from ..V_ion import (
    NonlocalProjectorOperator,
    apply_fourier_filters,
    build_nonlocal_projectors,
    check_core_density,
    ewald_alpha_z_energy,
    ewald_ion_ion_energy,
    ewald_local_ionic_potential,
    ionic_charge,
    load_pseudopotentials,
    normalize_density,
    superpose_atomic_density,
)
from ..V_ion.gspace_local import periodic_alpha_energy, periodic_local_ionic_potential
from ..V_xc import (
    NoncollinearXCResult,
    SpinPolarizedXCResult,
    SpinXCResult,
    XCResult,
    ca_lda,
    ca_lda_spin,
    ca_lda_spin_polarized,
    pbe,
    pbe_spin,
    noncollinear_xc,
    pbe_spin_polarized,
)
from ..V_xc.selection import (
    as_spin_polarized_result,
    evaluate_spin_xc,
    evaluate_unpolarized_xc,
)
from .single_point import _species_densities, initial_spin_densities
from .single_point import run_scf as _run_scf


class LazyGradient(Sequence):
    """The three centred first-derivative matrices ``d/dx, d/dy, d/dz`` of a grid, built on first use.

    The matrices of :func:`parsec_python.Laplacian.build_gradient`, element for element.  Only the k-point kinetic
    term ``-2 i k.grad``, band structures, meta-GGAs and the forces of wires and slabs read them; a Gamma-point run
    never builds them (2.0 M points of 2,744-atom diamond: 1.15 GB and 2 s of every preparation).  A sequence of
    the three CSR matrices for the code that reads them, safe to build from several threads.
    """

    def __init__(self, grid) -> None:
        self._grid = grid
        self._matrices: tuple | None = None
        self._lock = threading.Lock()

    def _built(self) -> tuple:
        if self._matrices is None:
            with self._lock:
                if self._matrices is None:
                    self._matrices = tuple(build_gradient(self._grid))
        return self._matrices

    @property
    def built(self) -> bool:
        """Whether the matrices exist (for tests and reports; reading them builds them)."""
        return self._matrices is not None

    def __getitem__(self, index):
        return self._built()[index]

    def __len__(self) -> int:
        return 3

    def __iter__(self):
        return iter(self._built())

    def __getstate__(self) -> dict:
        return {"grid": self._grid, "matrices": self._matrices}

    def __setstate__(self, state: dict) -> None:
        self._grid = state["grid"]
        self._matrices = state["matrices"]
        self._lock = threading.Lock()


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
    # d/dx, d/dy, d/dz (LazyGradient: built where a k-point, band or meta-GGA term reads them)
    gradient: Sequence[sp.csr_matrix]
    ionic_potential: np.ndarray
    nonlocal_operator: NonlocalProjectorOperator
    initial_density: np.ndarray
    core_density: np.ndarray
    ion_ion_energy: float
    atomic_reference_correction: float = 0.0
    alpha_z_energy: float = 0.0
    timings: PreparationTimings = field(default_factory=PreparationTimings)
    # LDA+U projectors with the images one cell away (upot.F90), or None.
    hubbard: HubbardProjectors | None = None
    # The Laplacian of a non-orthogonal cell (``GeneralPeriodicGrid``):
    # Fortran's ``pbc_grid_coefs`` data, for the report; ``None`` for the
    # orthorhombic grid.
    laplacian_coefficients: NonorthogonalLaplacianCoefficients | None = None
    # Starting densities of the two spin channels for the loop of
    # :func:`parsec_python.SCF.run_spin_scf` (forces, relaxation and molecular
    # dynamics of a spin-polarized cell, :func:`with_initial_spin_densities`);
    # ``None`` otherwise.
    initial_spin_densities: tuple[np.ndarray, np.ndarray] | None = None
    # Symmetrize_Charge_Density: the space group of ``input.space_group`` as
    # permutations of the grid points (symm_scalar of newrho.F90); ``None``
    # without it.  The SCF loops average every new density over it.
    density_symmetry: GridPointGroup | None = None

    @property
    def cell_atoms(self) -> tuple:
        """The atoms folded into the cell, in the order of ``atoms`` (see :func:`fold_atoms_into_cell`).

        ``atoms`` keeps the coordinates a relaxation or a dynamics run moves them to, which leave the cell; every
        builder of an image sum (local potential, projectors, densities, Ewald sums, forces) assumes atoms inside
        it and gets these.
        """
        if self.input.periodic_cell.dimensions == 2:
            # a slab of Andrew Grieder's slab_boundary: its atoms sit inside the box (|z| < R), which the 3-D fold
            # would move by the box height
            return tuple(self.atoms)
        # a wire or slab of the Fortran-derived routines (grid periodicity 1 or 2): folded along its periodic
        # vectors only; its cell has unit vectors along the confined directions, and a fold along them moved the
        # atoms of the k-point loops (and of the forces) away from those of the preparation (to 2026-11-08)
        periodicity = int(getattr(self.input.grid, "periodicity", 3))
        return fold_atoms_into_cell(self.atoms, self.input.periodic_cell.lattice_vectors, periodic_axes=periodicity)

    def hamiltonian(
        self,
        effective_potential: np.ndarray,
        hubbard: object | None = None,
        exchange: object | None = None,
        *,
        kinetic_potential: np.ndarray | None = None,
        gradient_flux=None,
    ) -> KohnShamHamiltonian:
        """Compose the current SCF Hamiltonian without forming a dense matrix.

        The supplied array is only the diagonal local field

        ``V_eff = V_ion,local + V_H + V_xc``.

        :class:`KohnShamHamiltonian` binds it to the already prepared kinetic
        and nonlocal terms, giving

        ``H = -nabla_FD^2 + diag(V_eff) + V_NL``,

        plus the Gamma-point LDA+U term ``hubbard`` where it is given, the
        exact-exchange term ``a_x V_ACE`` of a hybrid functional at the Gamma
        point (``exchange``, :mod:`parsec_python.SCF.hybrid`) and the
        orbital term ``-div(vtau grad)`` of a meta-GGA at the Gamma point
        (``kinetic_potential`` = ``vtau``, :mod:`parsec_python.V_xc.meta_gga`;
        ``gradient_flux`` = ``W`` of its orbital density gradient).  A
        periodic cell runs no meta-GGA hybrid (refused by the input), so the
        two terms never come together here.
        """
        if exchange is not None and kinetic_potential is not None:
            raise NotImplementedError("a meta-GGA hybrid of a periodic cell is not implemented")
        if kinetic_potential is not None:
            from ..V_xc.meta_gga import KineticPotentialOperator, grid_gradient

            return KohnShamHamiltonian(
                self.negative_laplacian,
                effective_potential,
                self.nonlocal_operator,
                hubbard,
                None,
                KineticPotentialOperator(grid_gradient(self.grid), kinetic_potential, gradient_flux=gradient_flux),
            )
        if exchange is not None:
            return KohnShamHamiltonian(
                self.negative_laplacian,
                effective_potential,
                self.nonlocal_operator,
                hubbard,
                exchange,
            )
        if hubbard is None:
            return KohnShamHamiltonian(
                self.negative_laplacian,
                effective_potential,
                self.nonlocal_operator,
            )
        return KohnShamHamiltonian(
            self.negative_laplacian,
            effective_potential,
            self.nonlocal_operator,
            hubbard,
        )

    def solve_hartree(
        self,
        density: np.ndarray,
        initial_potential: np.ndarray | None = None,
        *,
        raise_on_nonconvergence: bool = True,
    ) -> PeriodicHartreeResult:
        if self.input.periodic_cell.dimensions == 2:
            return solve_slab_hartree(
                density,
                self.grid,
                self.negative_laplacian,
                self.input.hartree,
                initial_potential,
                raise_on_nonconvergence=raise_on_nonconvergence,
            )
        return solve_periodic_hartree(
            density,
            self.grid,
            self.negative_laplacian,
            self.input.hartree,
            initial_potential,
            raise_on_nonconvergence=raise_on_nonconvergence,
        )

    def _zero_average(self, zero_average: bool | None) -> bool:
        """Whether an XC evaluation removes the average (Fortran ``potshift``)."""

        if zero_average is None:
            zero_average = getattr(self.input.scf, "zero_average_xc_potential", False)
        return bool(zero_average) and getattr(self.input.grid, "periodicity", 3) == 3

    def evaluate_xc(
        self,
        density: np.ndarray,
        tau: np.ndarray | None = None,
        density_gradient=None,
        *,
        zero_average: bool | None = None,
    ) -> XCResult:
        """Evaluate the selected XC functional including frozen NLCC.

        With ``SCFSettings.zero_average_xc_potential`` (``zero_average=None``
        follows it) the potential is shifted to a zero average, as
        ``potshift`` does in every ``rho_har`` call of a 3-D periodic cell; the
        energy is that of the functional either way.  A meta-GGA needs
        ``tau`` of the valence orbitals (:meth:`kinetic_energy_density`, or
        the k-point sum of :func:`parsec_python.SCF.kpoints.run_scf_kpoints`)
        and returns a :class:`~parsec_python.V_xc.meta_gga.MetaGGAXCResult`
        (``vtau`` is not shifted); ``density_gradient`` is ``g`` of the orbital
        form of the density gradient (three fields).
        """

        if getattr(self.input.scf, "is_meta_gga", False):
            from ..V_xc.meta_gga import meta_gga_evaluator, meta_gga_xc

            if tau is None:
                raise ValueError("a meta-GGA needs the kinetic-energy density of the orbitals")
            result = meta_gga_xc(
                meta_gga_evaluator(self.input.scf),
                density,
                tau,
                self.grid,
                self.core_density,
                negative_laplacian=self.negative_laplacian,
                name=self.input.scf.xc_functional,
                density_gradient=None if density_gradient is None else np.stack(density_gradient),
            )
        elif getattr(self.input.scf, "is_hybrid", False):
            # The semilocal part of a hybrid; the exact exchange is the operator of SCF/hybrid.py.
            from .hybrid import scf_semilocal_xc

            result = scf_semilocal_xc(self.input.scf, density, self.grid, self.core_density)
        else:
            # a device evaluator where one is installed and takes this system (set_periodic_xc_provider)
            result = _provided_xc(self, density)
            if result is None and self.input.scf.xc_functional == "ca":
                result = ca_lda(density, self.grid.volume_element, self.core_density)
            elif result is None and self.input.scf.xc_functional == "pbe":
                result = pbe(density, self.grid, self.core_density)
            elif result is None:
                result = evaluate_unpolarized_xc(
                    self.input.scf.xc_functional,
                    density,
                    self.grid,
                    self.core_density,
                    ionization_energy_difference=self.input.scf.ionization_energy_difference,
                )
        if not self._zero_average(zero_average):
            return result
        return replace(result, potential=_without_average(result.potential))

    def evaluate_xc_spin_polarized(
        self,
        density_up: np.ndarray,
        density_down: np.ndarray,
        *,
        zero_average: bool | None = None,
    ) -> SpinPolarizedXCResult:
        """Collinear spin-polarized XC (LSDA or PBE) including frozen NLCC.

        ``zero_average``: as :meth:`evaluate_xc`, each channel by its own
        average (``potshift`` per ``isp``).  A meta-GGA needs the ``tau`` of
        both channels: :meth:`evaluate_spin_xc`.
        """

        functional = self.input.scf.xc_functional
        if getattr(self.input.scf, "is_meta_gga", False):
            raise NotImplementedError(
                f"the meta-GGA {functional!r} needs the kinetic-energy densities of the orbitals: "
                "its spin loops call evaluate_spin_xc with tau_up and tau_down"
            )
        if getattr(self.input.scf, "is_hybrid", False):
            from .hybrid import scf_semilocal_spin_xc

            result = as_spin_polarized_result(
                scf_semilocal_spin_xc(self.input.scf, density_up, density_down, self.grid, self.core_density)
            )
        else:
            result = _provided_spin_xc(self, "spin_polarized", density_up, density_down)
        if result is None:
            if functional == "ca":
                result = ca_lda_spin_polarized(
                    density_up,
                    density_down,
                    self.grid.volume_element,
                    self.core_density,
                )
            elif functional == "pbe":
                result = pbe_spin_polarized(
                    density_up, density_down, self.grid, self.core_density
                )
            else:
                result = as_spin_polarized_result(
                    evaluate_spin_xc(
                        functional,
                        density_up,
                        density_down,
                        self.grid,
                        self.core_density,
                        ionization_energy_difference=self.input.scf.ionization_energy_difference,
                    )
                )
        if not self._zero_average(zero_average):
            return result
        return replace(
            result,
            potential_up=_without_average(result.potential_up),
            potential_down=_without_average(result.potential_down),
        )

    def evaluate_spin_xc(
        self,
        density_up: np.ndarray,
        density_down: np.ndarray,
        tau_up: np.ndarray | None = None,
        tau_down: np.ndarray | None = None,
        density_gradients=None,
        *,
        zero_average: bool | None = None,
    ):
        """The spin XC of :meth:`PreparedSinglePointSystem.evaluate_spin_xc`.

        What :func:`parsec_python.SCF.run_spin_scf` evaluates; each channel
        sees its valence density plus half of the frozen NLCC density.
        ``zero_average``: as :meth:`evaluate_xc_spin_polarized`.  A meta-GGA
        needs the ``tau`` of both channels and returns a
        :class:`~parsec_python.V_xc.meta_gga.MetaGGASpinXCResult`.
        """

        functional = self.input.scf.xc_functional
        if getattr(self.input.scf, "is_meta_gga", False):
            from ..V_xc.meta_gga import meta_gga_evaluator, meta_gga_xc_spin

            if tau_up is None or tau_down is None:
                raise ValueError("a meta-GGA needs the kinetic-energy densities of both channels")
            result = meta_gga_xc_spin(
                meta_gga_evaluator(self.input.scf),
                density_up,
                density_down,
                tau_up,
                tau_down,
                self.grid,
                self.core_density,
                negative_laplacian=self.negative_laplacian,
                name=functional,
                density_gradients=None
                if density_gradients is None
                else tuple(np.stack(gradient) for gradient in density_gradients),
            )
        elif getattr(self.input.scf, "is_hybrid", False):
            from .hybrid import scf_semilocal_spin_xc

            result = scf_semilocal_spin_xc(self.input.scf, density_up, density_down, self.grid, self.core_density)
        else:
            result = _provided_spin_xc(self, "spin", density_up, density_down)
        if result is None:
            if functional == "ca":
                result = ca_lda_spin(
                    density_up, density_down, self.grid.volume_element, self.core_density
                )
            elif functional == "pbe":
                result = pbe_spin(density_up, density_down, self.grid, self.core_density)
            else:
                result = evaluate_spin_xc(
                    functional,
                    density_up,
                    density_down,
                    self.grid,
                    self.core_density,
                    ionization_energy_difference=self.input.scf.ionization_energy_difference,
                )
        if not self._zero_average(zero_average):
            return result
        return replace(
            result,
            potential_up=_without_average(result.potential_up),
            potential_down=_without_average(result.potential_down),
        )

    def evaluate_xc_noncollinear(
        self, density: np.ndarray, magnetization: np.ndarray
    ) -> NoncollinearXCResult:
        """Non-collinear XC (``V_avg`` and ``B_xc``) from ``n`` and ``m(r)``."""

        if self._zero_average(None):
            raise NotImplementedError(
                "zero_average_xc_potential (Fortran potshift) is not implemented for "
                "non-collinear magnetism"
            )
        return noncollinear_xc(density, magnetization, self.evaluate_xc_spin_polarized)

    # ----- the hooks of a meta-GGA loop (parsec_python.SCF.meta_gga) -----

    def kinetic_energy_density(self, wavefunctions, occupations, degeneracy: float = 2.0, gradient: bool = False):
        """``tau`` (Hartree) of the Euclidean-normalized real grid orbitals of a Gamma-point loop.

        :func:`parsec_python.V_xc.meta_gga.kinetic_energy_density` with the
        gradient of this cell (wrapped on the periodic axes, zero outside a
        slab or wire); with ``gradient`` also the orbital form of the density
        gradient, ``(tau, (g_x, g_y, g_z))``
        (:func:`parsec_python.V_xc.meta_gga.orbital_density_gradient`).  The
        k-point loops form both from the Bloch factors
        (:func:`parsec_python.SCF.meta_gga_periodic.bloch_orbital_fields`).
        """

        from ..V_xc.meta_gga import grid_gradient, kinetic_energy_density, orbital_density_gradient

        operators = grid_gradient(self.grid)
        tau = kinetic_energy_density(
            wavefunctions, occupations, operators, self.grid.volume_element, degeneracy=degeneracy
        )
        if not gradient:
            return tau
        values = orbital_density_gradient(
            wavefunctions, occupations, operators, self.grid.volume_element, degeneracy=degeneracy
        )
        if self.density_symmetry is not None:
            # Symmetrize_Charge_Density: the loop averages tau with the density; g is a vector field.
            from .meta_gga_periodic import symmetrized_gradient

            values = symmetrized_gradient(values, self.density_symmetry, self.input.periodic_cell.lattice_vectors)
        return tau, tuple(values)

    def evaluate_meta_gga_start_xc(self, density: np.ndarray) -> XCResult:
        """PBE of the start density of a meta-GGA loop (PARSEC 1.5's first iteration), with the zero-average shift."""

        result = pbe(density, self.grid, self.core_density)
        if not self._zero_average(None):
            return result
        return replace(result, potential=_without_average(result.potential))

    def evaluate_meta_gga_start_spin_xc(self, density_up: np.ndarray, density_down: np.ndarray):
        """Spin PBE of the start densities of a meta-GGA loop, with the zero-average shift."""

        result = pbe_spin(density_up, density_down, self.grid, self.core_density)
        if not self._zero_average(None):
            return result
        return replace(
            result,
            potential_up=_without_average(result.potential_up),
            potential_down=_without_average(result.potential_down),
        )


# A faster evaluator of the spin functionals of a periodic cell (a CUDA
# device), or None.
_spin_xc_provider = None
_xc_provider = None


def set_periodic_xc_provider(provider=None) -> None:
    """Install (or, with ``None``, remove) a faster unpolarized XC of periodic cells.

    ``provider(system, density)`` returns what
    :meth:`PeriodicPreparedSinglePointSystem.evaluate_xc` evaluates for a
    semilocal functional (not a meta-GGA or a hybrid) before the zero-average
    shift, or ``None`` for a system or functional it does not take (the
    readable functional then runs).  The performance counterpart lives in
    ``parsec_python.acceleration.V_xc.periodic_xc`` (the replicated drivers of
    an MPI run); importing :mod:`parsec_python.acceleration` installs it.
    """

    global _xc_provider
    _xc_provider = provider


def _provided_xc(system, density):
    if _xc_provider is None:
        return None
    return _xc_provider(system, density)


def set_periodic_spin_xc_provider(provider=None) -> None:
    """Install (or, with ``None``, remove) a faster spin XC of periodic cells.

    ``provider(system, kind, density_up, density_down)`` returns what
    :meth:`PeriodicPreparedSinglePointSystem.evaluate_xc_spin_polarized`
    (``kind="spin_polarized"``) or :meth:`~PeriodicPreparedSinglePointSystem.evaluate_spin_xc`
    (``kind="spin"``) evaluate before the zero-average shift, or ``None`` for
    a system or functional it does not take (the readable functional then
    runs).  The performance counterpart lives in
    ``parsec_python.acceleration.V_xc.periodic_xc``; importing
    :mod:`parsec_python.acceleration` installs it.
    """

    global _spin_xc_provider
    _spin_xc_provider = provider


def _provided_spin_xc(system, kind: str, density_up, density_down):
    if _spin_xc_provider is None:
        return None
    return _spin_xc_provider(system, kind, density_up, density_down)


def _without_average(potential: np.ndarray) -> np.ndarray:
    """``potential - <potential>`` (``potshift.f90``: ``sum * hcub / vcell``).

    Every grid point of a periodic cell carries the volume ``Omega/N``, so the
    average is the plain mean of the full grid.
    """

    values = np.asarray(potential, dtype=np.float64)
    return values - float(np.mean(values))


def atom_outside_cell(lattice_vectors: np.ndarray, atoms) -> Atom | None:
    """The first atom outside the cell (Andrew Grieder's test: ``[0, L)`` of an orthorhombic cell, fractional
    coordinates in ``[0, 1)`` within 1e-12 of an oblique one), ``None`` when all are inside."""

    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
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
            return atom
    return None


def _prepare_fortran_periodic_single_point(
    problem: SinglePointInput,
) -> PeriodicPreparedSinglePointSystem:
    """Build every static periodic component, but do not enter the SCF loop (Fortran-derived routines).

    :func:`prepare_periodic_single_point` of the Fortran parity round, for ``Use_Fortran_Routines``
    (``problem.grid.fortran_routines``) and for wires and slabs.

    ``problem.periodic_cell`` must be set and ``problem.grid`` must be a
    :class:`~..models.PeriodicGridSettings` (enforced by
    ``SinglePointInput.__post_init__``).  Atom positions are Cartesian bohr;
    this path does not recenter geometry the way the isolated path does.
    An atom outside the cell (fractional coordinates outside ``[0, 1)``) is
    translated into it by lattice vectors (:func:`~..Grid.cell.wrap_into_cell`),
    which changes no periodic sum; ``system.atoms`` holds the translated
    positions.

    A cell the orthorhombic grid does not take (non-orthogonal lattice
    vectors, or sides without one common spacing) gets Fortran's
    :class:`~..Grid.pbc.GeneralPeriodicGrid`: its Laplacian has the auxiliary
    directions of ``pbc_grid_coefs`` (``laplacian_coefficients``), its
    gradient and the GGA work along the lattice directions, and its Hartree
    equation is solved by FFT.
    """
    if problem.periodic_cell is None:
        raise ValueError("prepare_periodic_single_point requires problem.periodic_cell")
    if getattr(problem.grid, "periodicity", 3) in (1, 2):
        # Boundary_Conditions wire or slab (parsec_python.SCF.wire_slab).
        from .wire_slab import prepare_wire_slab_single_point

        return prepare_wire_slab_single_point(problem)
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
    # Alpha_Filter/Core_Filter: inipbc.f90 filters the radial tables with the
    # final grid spacing grid%step(1) (fourier_f.f90); a general grid has one
    # step per lattice direction.
    filter_spacing = (
        float(grid.spacings[0])
        if getattr(grid, "is_general_periodic", False)
        else grid.spacing
    )
    pseudopotentials = apply_fourier_filters(
        pseudopotentials, problem.pseudopotentials, filter_spacing
    )
    # Every periodic sum runs over lattice images, so an atom outside the
    # cell is translated into it by lattice vectors (Fortran PARSEC takes
    # such coordinates as they are); atoms inside keep their coordinates.
    placed = []
    for atom in atoms:
        wrapped = wrap_into_cell(atom.position, problem.periodic_cell.lattice_vectors)
        placed.append(atom if wrapped is atom.position else replace(atom, position=wrapped))
    atoms = tuple(placed)
    laplacian_coefficients = (
        grid_laplacian_coefficients(grid)
        if getattr(grid, "is_general_periodic", False)
        else None
    )
    # Symmetrize_Charge_Density: the space group (the identity where the
    # input has none) as permutations of the grid points.
    density_symmetry = None
    if problem.scf.symmetrize_density:
        group = getattr(problem, "space_group", None) or identity_space_group()
        density_symmetry = periodic_grid_symmetry(grid, group)

    stage_start = time.perf_counter()
    negative_laplacian = build_negative_laplacian(grid)
    # Only needed for k-point sampling's -2i*k.grad kinetic term
    # (Hamiltonian.kpoint_operator), bands and meta-GGAs: built when one
    # reads it, so that a Gamma-point run does not hold it.
    gradient = LazyGradient(grid)
    finite_difference_seconds = time.perf_counter() - stage_start

    # 3. Local and Kleinman--Bylander nonlocal ionic terms.
    stage_start = time.perf_counter()
    # The real-space Ewald split, or Fortran's G-space potential
    # (PeriodicGridSettings.local_potential).
    ionic_potential = periodic_local_ionic_potential(
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
    hubbard = build_hubbard_projectors(
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
    core_density = check_core_density(core_density)
    core_density_seconds = time.perf_counter() - stage_start

    # 5. Geometry-only ion--ion contribution to the total energy.
    stage_start = time.perf_counter()
    repulsion = ewald_ion_ion_energy(
        atoms, pseudopotentials, problem.periodic_cell.lattice_vectors
    )
    alpha_z_energy = periodic_alpha_energy(
        problem.grid,
        atoms,
        pseudopotentials,
        problem.pseudopotentials,
        problem.periodic_cell.lattice_vectors,
        electron_count,
    )
    if getattr(problem.grid, "local_potential", "real_space") == "real_space":
        # PARSEC 1.5's zero-average potential and its alpha term (the G-space potential has no G = 0 term)
        ionic_potential, alpha_z_energy = zero_average_local_potential(
            ionic_potential,
            atoms,
            pseudopotentials,
            problem.pseudopotentials,
            problem.periodic_cell.lattice_vectors,
            electron_count,
            alpha_term=getattr(problem.grid, "alpha_term", "analytic"),
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
        gradient=gradient,
        ionic_potential=ionic_potential,
        nonlocal_operator=nonlocal_operator,
        initial_density=initial_density,
        core_density=core_density,
        ion_ion_energy=repulsion,
        atomic_reference_correction=atomic_reference_correction,
        alpha_z_energy=alpha_z_energy,
        timings=preparation_timings,
        hubbard=hubbard,
        laplacian_coefficients=laplacian_coefficients,
        density_symmetry=density_symmetry,
    )


def wrapped_atoms(atoms, lattice_vectors) -> tuple:
    """The atoms translated into the cell, as :func:`prepare_periodic_single_point` does.

    An atom inside the cell keeps its object; one outside is moved by lattice
    vectors (``back_to_cell`` of ``initial.f90``, which PARSEC calls after
    every ``domove`` and ``moldyn`` of a periodic system; Fortran's cell is
    centred on the origin, this one is ``[0, 1)`` in fractional coordinates).
    """

    placed = []
    for atom in atoms:
        wrapped = wrap_into_cell(atom.position, lattice_vectors)
        placed.append(atom if wrapped is atom.position else replace(atom, position=wrapped))
    return tuple(placed)


def periodic_spin_start(system, atoms=None) -> tuple[np.ndarray, np.ndarray]:
    """Starting densities of the two channels (``initchrg.f90``) of a periodic cell.

    :func:`parsec_python.SCF.initial_spin_densities` of the superposed atomic
    densities with their lattice images: what the isolated preparation builds
    for :func:`parsec_python.SCF.run_spin_scf`.
    """

    from .single_point import initial_spin_densities

    problem = system.input
    atoms = system.atoms if atoms is None else tuple(atoms)
    lattice = problem.periodic_cell.lattice_vectors
    specifications = problem.pseudopotentials

    def superposed(selection):
        return superpose_atomic_density(
            system.grid, selection, system.pseudopotentials, specifications, lattice
        )

    total = superposed(atoms)
    symbols = list(dict.fromkeys(atom.symbol for atom in atoms))
    polarizations = {symbol: specifications[symbol].initial_spin_polarization for symbol in symbols}
    if len(set(polarizations.values())) == 1:
        species = {symbols[0]: total}
    else:
        species = {
            symbol: superposed(tuple(atom for atom in atoms if atom.symbol == symbol))
            for symbol in symbols
        }
    scale = (
        system.electron_count / system.grid.integrate(total)
        if problem.scf.normalize_initial_density
        else 1.0
    )
    return initial_spin_densities(total, species, polarizations, scale)


def with_initial_spin_densities(system: PeriodicPreparedSinglePointSystem):
    """``system`` with the starting densities :func:`parsec_python.SCF.run_spin_scf` reads.

    Unchanged when they are there already or the calculation is not spin
    polarized.  The single-point loop of a periodic spin-polarized cell
    (:func:`parsec_python.SCF.run_scf_spin_polarized`) does not read them, so
    the preparation does not build them.
    """

    if not system.input.scf.spin_polarized or system.initial_spin_densities is not None:
        return system
    return replace(system, initial_spin_densities=periodic_spin_start(system))


def fortran_update_periodic_geometry(
    system: PeriodicPreparedSinglePointSystem, atoms
) -> PeriodicPreparedSinglePointSystem:
    """Return ``system`` with its atoms at new positions on the same grid (Fortran-derived routines).

    ``update_periodic_geometry`` of the Fortran parity round (:func:`update_periodic_geometry` hands the systems of
    ``Use_Fortran_Routines`` here).

    The periodic counterpart of :func:`parsec_python.SCF.update_geometry`:
    what ``parsec.F90`` (l.1930-2046) repeats after ``domove``/``moldyn`` for
    a bulk cell, ``back_to_cell`` (atoms that left the cell are translated
    back), ``nonloc`` (projectors with their images), ``upot`` (LDA+U),
    ``corecd`` (core density), ``ionpbc`` (local potential) and
    ``ewald_sum`` (ion-ion energy).  Grid, kinetic operator, gradient,
    pseudopotentials and electron count are shared with ``system``.  The
    superposed atomic density of the new positions is built as well (a cold
    start and ``Potential_Extrapolation`` read it), and so are the starting
    spin densities where ``system`` has them.  ``system.input.atoms`` holds the
    positions as given (the integrator and the optimizer keep them
    unwrapped), ``atoms`` of the result the translated ones.
    """

    started = time.perf_counter()
    atoms = tuple(atoms)
    if len(atoms) != len(system.atoms) or any(
        new.symbol != old.symbol for new, old in zip(atoms, system.atoms)
    ):
        raise ValueError("moved atoms must keep their number, order and species")
    problem = replace(system.input, atoms=atoms)
    lattice = problem.periodic_cell.lattice_vectors
    placed = wrapped_atoms(atoms, lattice)
    grid = system.grid
    pseudopotentials = system.pseudopotentials
    specifications = problem.pseudopotentials

    stage_start = time.perf_counter()
    ionic_potential = periodic_local_ionic_potential(
        grid, placed, pseudopotentials, specifications, lattice
    )
    local_ionic_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    nonlocal_operator = build_nonlocal_projectors(
        grid, placed, pseudopotentials, specifications, lattice
    )
    hubbard = build_hubbard_projectors(
        grid, placed, pseudopotentials, specifications, lattice
    )
    nonlocal_ionic_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    initial_density = superpose_atomic_density(
        grid, placed, pseudopotentials, specifications, lattice
    )
    if problem.scf.normalize_initial_density:
        initial_density = normalize_density(initial_density, grid, system.electron_count)
    initial_density_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    core_density = check_core_density(
        superpose_atomic_density(
            grid, placed, pseudopotentials, specifications, lattice, core=True
        )
    )
    core_density_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    repulsion = ewald_ion_ion_energy(placed, pseudopotentials, lattice)
    alpha_z_energy = periodic_alpha_energy(
        problem.grid, placed, pseudopotentials, specifications, lattice, system.electron_count
    )
    if getattr(problem.grid, "local_potential", "real_space") == "real_space":
        # PARSEC 1.5's zero-average potential and its alpha term (zero_average_local_potential)
        ionic_potential, alpha_z_energy = zero_average_local_potential(
            ionic_potential,
            placed,
            pseudopotentials,
            specifications,
            lattice,
            system.electron_count,
            alpha_term=getattr(problem.grid, "alpha_term", "analytic"),
        )
    ion_ion_seconds = time.perf_counter() - stage_start
    moved = replace(
        system,
        input=problem,
        atoms=placed,
        ionic_potential=ionic_potential,
        nonlocal_operator=nonlocal_operator,
        initial_density=initial_density,
        core_density=core_density,
        ion_ion_energy=repulsion,
        alpha_z_energy=alpha_z_energy,
        hubbard=hubbard,
        initial_spin_densities=None,
    )
    if system.initial_spin_densities is not None:
        moved = replace(moved, initial_spin_densities=periodic_spin_start(moved))
    return replace(
        moved,
        timings=PreparationTimings(
            local_ionic_seconds=local_ionic_seconds,
            nonlocal_ionic_seconds=nonlocal_ionic_seconds,
            initial_density_seconds=initial_density_seconds,
            core_density_seconds=core_density_seconds,
            ion_ion_seconds=ion_ion_seconds,
            total_seconds=time.perf_counter() - started,
        ),
    )


def fold_atoms_into_cell(atoms, lattice_vectors: np.ndarray, *, periodic_axes: int = 3) -> tuple:
    """Atoms with their positions moved by lattice vectors into the cell (fractional coordinates in ``[0, 1)``).

    A periodic image of an atom is the same atom: the energy and the forces do not depend on which one the
    coordinates name, but the builders of image sums are written for atoms inside the cell.  Atoms already inside are
    returned as they are, bit for bit.  ``periodic_axes`` (1 for a wire, 2 for a slab) folds along the first lattice
    vectors only and leaves the confined coordinates as they are.
    """

    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    inverse = np.linalg.inv(lattice)
    axes = int(periodic_axes)
    folded = []
    for atom in atoms:
        position = np.asarray(atom.position, dtype=np.float64)
        fraction = (position @ inverse)[:axes]
        inside = np.all(fraction >= -1.0e-12) and np.all(fraction < 1.0 + 1.0e-12)
        if inside:
            folded.append(atom)
            continue
        if axes == 3:
            wrapped = fraction - np.floor(fraction)
            folded.append(replace(atom, position=wrapped @ lattice))
        else:
            folded.append(replace(atom, position=position - np.floor(fraction) @ lattice[:axes]))
    return tuple(folded)


def zero_average_local_potential(
    potential: np.ndarray,
    atoms,
    pseudopotentials,
    specifications,
    lattice_vectors: np.ndarray,
    electron_count: float,
    *,
    alpha_term: str = "analytic",
) -> tuple[np.ndarray, float]:
    """``(V - c, E_alpha)`` of the real-space Ewald split: PARSEC 1.5's convention for a 3-D cell (Ry).

    Fortran builds the local potential of a bulk cell in G space without its ``G = 0`` term (``ionpbc.f90``:
    ``vionz(1) = 0``), a zero average, and the energy carries ``E_alpha = (N_e/Omega) sum_a alpha_a``,
    ``alpha_a = int (V_a + 2 Z/r) d^3r`` (``pbc%ealpha``;
    :func:`parsec_python.V_ion.gspace_local.gspace_alpha_energy`).  The real-space split
    (:func:`parsec_python.V_ion.ewald.ewald_local_ionic_potential`) keeps the cell average of its split,
    ``c = sum_a (alpha_a - 2 pi Z_a/eta^2)/V`` at ``eta = sqrt(pi)/V^(1/3)`` (``-2 Z_tot/V^(1/3)`` grows with the
    cell: -608 Ry for 4,096 carbon atoms of diamond), which its alpha term ``N_e 2 pi Z_tot/(eta^2 V)``
    (:func:`parsec_python.V_ion.ewald.ewald_alpha_z_energy`) cancelled in the energy up to 2026-11-08.  The
    preparations of a bulk cell now remove ``c`` and take Fortran's ``E_alpha``: the band energy loses ``N_e c``
    (``2 sum f = N_e``) and ``E_alpha - N_e 2 pi Z_tot/(eta^2 V) = N_e c``, so the total energy, the density and the
    forces are unchanged (``c`` does not depend on the positions), and the eigenvalues and the Fermi level lose
    ``c``.  The potential keeps the difference between the grid sum of the short-ranged pieces and their integral
    (4e-4 Ry for carbon at 18 points per 3.567 A).
    """

    from ..V_ion.ewald import _ewald_cutoffs
    from ..V_ion.gspace_local import alpha_coefficient, gspace_alpha_energy

    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    volume = float(abs(np.linalg.det(lattice)))
    eta, _, _ = _ewald_cutoffs(volume, None, 1.0e-14)
    coefficients: dict[str, float] = {}
    average = 0.0
    for atom in atoms:
        if atom.symbol not in coefficients:
            potential_of = pseudopotentials[atom.symbol]
            coefficients[atom.symbol] = alpha_coefficient(
                potential_of, specifications[atom.symbol].local_angular_momentum, alpha_term
            ) - 2.0 * np.pi * float(potential_of.ionic_charge) / eta**2
        average += coefficients[atom.symbol]
    average /= volume
    energy = gspace_alpha_energy(
        atoms, pseudopotentials, specifications, lattice, electron_count, method=alpha_term
    )
    return np.asarray(potential, dtype=np.float64) - average, energy


def _geometry_terms(
    problem: SinglePointInput,
    grid,
    atoms,
    pseudopotentials,
    electron_count: float,
    *,
    negative_laplacian=None,
) -> dict:
    """Everything of a periodic system that depends on the atom positions, for atoms inside the cell.

    The local ionic potential, the projectors with their images, the atomic starting density and the core density, the
    Ewald ion-ion energy and the alpha-Z term, with the seconds of each stage (``PreparationTimings`` fields).  A slab
    (``periodic_cell.dimensions == 2``, Andrew Grieder's slab_boundary) has the Gaussian-compensated local potential
    of :mod:`parsec_python.V_ion.slab` (which needs ``negative_laplacian``), images along its in-plane vectors only and
    the 2-D Ewald energy.
    """

    lattice_vectors = problem.periodic_cell.lattice_vectors
    dimensions = problem.periodic_cell.dimensions
    stage_start = time.perf_counter()
    if dimensions == 2:
        ionic_potential = slab_local_ionic_potential(
            grid, atoms, pseudopotentials, problem.pseudopotentials, negative_laplacian, problem.hartree
        )
    else:
        ionic_potential = ewald_local_ionic_potential(
            grid, atoms, pseudopotentials, problem.pseudopotentials, lattice_vectors
        )
    local_ionic_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    nonlocal_operator = build_nonlocal_projectors(
        grid, atoms, pseudopotentials, problem.pseudopotentials, lattice_vectors, periodic_dimensions=dimensions
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
        grid, atoms, pseudopotentials, problem.pseudopotentials, lattice_vectors, periodic_dimensions=dimensions
    )
    spin_densities = None
    if problem.scf.spin_polarized:
        # initchrg.f90: the two channels carry the factor that normalizes the total density (see initial_spin_densities)
        def superpose(grid_, atoms_, pseudopotentials_, specifications_):
            return superpose_atomic_density(
                grid_, atoms_, pseudopotentials_, specifications_, lattice_vectors, periodic_dimensions=dimensions
            )

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
        grid,
        atoms,
        pseudopotentials,
        problem.pseudopotentials,
        lattice_vectors,
        core=True,
        periodic_dimensions=dimensions,
    )
    core_density_seconds = time.perf_counter() - stage_start

    # Geometry-only ion--ion contribution to the total energy.
    stage_start = time.perf_counter()
    if dimensions == 2:
        repulsion = slab_ewald_energy(atoms, pseudopotentials, lattice_vectors)
        alpha_z_energy = 0.0  # the slab's local potential is summed in real space: no G = 0 correction
    else:
        repulsion = ewald_ion_ion_energy(atoms, pseudopotentials, lattice_vectors)
        # PARSEC 1.5's zero-average potential and its alpha term
        ionic_potential, alpha_z_energy = zero_average_local_potential(
            ionic_potential, atoms, pseudopotentials, problem.pseudopotentials, lattice_vectors, electron_count
        )
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

    A system of the Fortran-derived routines goes to :func:`fortran_update_periodic_geometry`.
    """

    if uses_fortran_routines(system.grid):
        return fortran_update_periodic_geometry(system, atoms)
    started = time.perf_counter()
    atoms = tuple(atoms)
    if len(atoms) != len(system.atoms) or any(new.symbol != old.symbol for new, old in zip(atoms, system.atoms)):
        raise ValueError("moved atoms must keep their number, order and species")
    problem = replace(system.input, atoms=atoms)
    folded = fold_atoms_into_cell(atoms, problem.periodic_cell.lattice_vectors)
    terms = _geometry_terms(
        problem,
        system.grid,
        folded,
        system.pseudopotentials,
        system.electron_count,
        negative_laplacian=system.negative_laplacian,
    )
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

    These are Andrew Grieder's routines, the default; ``problem.grid.fortran_routines``
    (``Use_Fortran_Routines``) selects :func:`_prepare_fortran_periodic_single_point`.
    """
    if problem.periodic_cell is None:
        raise ValueError("prepare_periodic_single_point requires problem.periodic_cell")
    if getattr(problem.grid, "fortran_routines", False) or getattr(problem.grid, "periodicity", 3) in (1, 2):
        # Use_Fortran_Routines (parsec_python.Input.routines), and the wires and slabs of those routines
        return _prepare_fortran_periodic_single_point(problem)
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
    dimensions = problem.periodic_cell.dimensions
    orthorhombic = np.allclose(lattice_vectors, np.diag(np.diag(lattice_vectors)), atol=1.0e-10)
    side_lengths = np.diag(lattice_vectors)
    inverse_lattice = None if orthorhombic else np.linalg.inv(lattice_vectors)
    for atom in atoms:
        position = np.asarray(atom.position, dtype=float)
        if dimensions == 2:
            # in-plane fractional coordinates in [0, 1); along z the atom must sit inside the box
            fraction = position[:2] @ np.linalg.inv(lattice_vectors[:2, :2])
            outside = np.any(fraction < -1.0e-12) or np.any(fraction >= 1.0 + 1.0e-12)
            outside = outside or abs(position[2]) >= 0.5 * lattice_vectors[2, 2]
            where = "the slab cell (in-plane fractional coordinates in [0, 1), |z| < R)"
        elif orthorhombic:
            outside = np.any(position < 0.0) or np.any(position >= side_lengths)
            where = f"cell [0, {side_lengths.tolist()})"
        else:
            fraction = position @ inverse_lattice
            outside = np.any(fraction < -1.0e-12) or np.any(fraction >= 1.0 + 1.0e-12)
            where = "the cell (fractional coordinates in [0, 1))"
        if outside:
            raise ValueError(
                f"atom {atom.symbol} at {position} lies outside the periodic "
                f"{where}; this path does not "
                "recenter geometry the way the isolated path does"
            )

    stage_start = time.perf_counter()
    negative_laplacian = build_negative_laplacian(grid)
    # Only needed for k-point sampling's -2i*k.grad kinetic term
    # (Hamiltonian.kpoint_operator), bands and meta-GGAs: built when one
    # reads it, so that a Gamma-point run does not hold it.
    gradient = LazyGradient(grid)
    finite_difference_seconds = time.perf_counter() - stage_start

    # 3. The terms that depend on the positions: local and Kleinman--Bylander nonlocal ionic terms, the starting and
    # core densities, the ion--ion energy.
    terms = _geometry_terms(
        problem, grid, atoms, pseudopotentials, electron_count, negative_laplacian=negative_laplacian
    )
    atomic_reference_correction = float(
        sum(
            problem.pseudopotentials[atom.symbol].atomic_energy_correction
            for atom in atoms
        )
    )

    # Symmetrize_Charge_Density of the irreducible wedge of a Monkhorst_Pack_Shift mesh: the operations that map
    # this grid onto itself (the parser's Input.parsec_input._default_routines_wedge); without a group these
    # routines average nothing, as before.
    group = getattr(problem, "space_group", None)
    density_symmetry = (
        periodic_grid_symmetry(grid, group)
        if problem.scf.symmetrize_density and group is not None and group.order > 1
        else None
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
        density_symmetry=density_symmetry,
    )


__all__ = [
    "LazyGradient",
    "PeriodicPreparedSinglePointSystem",
    "atom_outside_cell",
    "fold_atoms_into_cell",
    "zero_average_local_potential",
    "fortran_update_periodic_geometry",
    "periodic_spin_start",
    "prepare_periodic_single_point",
    "update_periodic_geometry",
    "set_periodic_spin_xc_provider",
    "set_periodic_xc_provider",
    "with_initial_spin_densities",
    "wrapped_atoms",
]
