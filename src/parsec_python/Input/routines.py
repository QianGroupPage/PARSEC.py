"""Which routines a calculation runs: Andrew Grieder's or the Fortran-derived ones.

Two sets of routines exist for the periodic cells (grid rule and shift, oblique-cell operator, local potential,
symmetry, forces, relaxation and dynamics, slabs) and for the forces and the ionic loops of spinor orbitals:

* Andrew Grieder's (qian/main 5101aa94, qian/pbc_forces 83067af9, qian/slab_boundary 3d3e4336);
* the Fortran-derived routines of the Fortran parity round.

The input parameter ``Use_Fortran_Routines`` chooses:

* not given: the default of each feature - Andrew Grieder's routines, except for the features whose default is
  the Fortran-derived routines (:data:`FORTRAN_DEFAULT_FEATURES`): slabs, a shifted Monkhorst-Pack grid of a bulk
  cell (``Monkhorst_Pack_Shift``), the forces, relaxation and molecular dynamics of a bulk cell at the Gamma point
  on a grid both sets of routines share with Cubic_Spline species, and a symmetric bulk cell whose space group sets Fortran's grid (:func:`bulk_default_features`; the user's
  decisions of 2 and 4 November 2026);
* ``.false.``: Andrew Grieder's routines wherever his code accepts the input (his slab, his shifted mesh too);
* ``.true.``: the Fortran-derived routines.

An input that Andrew Grieder's routines do not compute (a keyword or a value they do not read, a geometry they
refuse) runs on the Fortran-derived routines by itself, and the reason is printed with a warning.
:func:`decide_routines` makes the choice once per run, from the input's keywords; the parser adds the reasons it
finds in the values and the features whose default is the Fortran-derived routines
(``parsec_python.Input.parsec_input``) and records the result on the translation
(``ParsecInputTranslation.routines``) and on the periodic grid settings (``PeriodicGridSettings.fortran_routines``),
which every periodic routine reads.  Both CLIs follow it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

#: The labels of the input that the default routines read (Andrew Grieder's parser: ``species_keys``,
#: ``accepted_global`` and ``block_keys`` of ``Input/parsec_input.py`` on qian/pbc_forces and
#: qian/slab_boundary, which agree).  Any other label is read by the Fortran-derived routines only.
DEFAULT_SPECIES_KEYS = frozenset(
    {
        "atom_type",
        "element_symbol",
        "pseudopotential_format",
        "local_component",
        "read_vcd",
        "cubic_spline",
        "so_psp",
        "atomic_energy_correction",
        "move_flag",
        "potential_num",
        "core_cutoff_radius",
        "initial_spin_polarization",
    }
)
DEFAULT_BLOCK_KEYS = frozenset(
    {
        "atom_coord",
        "domain_shape_parameters",
        "cell_shape",
        "monkhorst_pack_grid",
        "monkhorst_pack_shift",
        "initial_ncl_moment",
        "kpoint_list",
        "bandstruc",
        "electron_per_orbital",
    }
)
DEFAULT_GLOBAL_KEYS = frozenset(
    {
        "restart_run",
        "relax_restart",
        "old_pseudopotential_format",
        "periodic_system",
        "boundary_conditions",
        "cluster_domain_shape",
        "boundary_sphere_radius",
        "grid_spacing",
        "coordinate_unit",
        "ignore_symmetry",
        "expansion_order",
        "states_num",
        "net_charges",
        "fermi_temp",
        "spin_polarization",
        "scf_so",
        "so_from_scratch",
        "non_collinear_magnetism",
        "kpoint_method",
        "kpoint_unit",
        "kpoint_file",
        "bandstruc_points",
        "max_iter",
        "convergence_criterion",
        "use_plain_sre",
        "diag_tolerance",
        "eigensolver",
        "chebdav_degree",
        "ff_maxiter",
        "matvec_blocksize",
        "chebyshev_degree",
        "chebyshev_degree_delta",
        "subspace_buffer_size",
        "mixing_method",
        "mixing_param",
        "memory_param",
        "restart_mixing",
        "mixing_safeguard",
        "mixing_step_limit",
        "mixing_growth_trigger",
        "mixing_backoff",
        "solver_lpole",
        "full_hartree",
        "hartree_boundary_tolerance",
        "hartree_atomic_tail",
        "domain_energy_tolerance",
        "atom_types_num",
        "correlation_type",
        "electric_field",
        "output_all_states",
        "output_level",
        "old_interpolation_format",
        "double_grid_order",
        "minimization",
        "movement_num",
        "skip_force",
        "total_atom_num",
        "ion_energy_diff",
        "force_min",
        "max_step",
        "min_step",
        "bfgs_number_corr",
        "relaxation_scf_upscale",
        "molecular_dynamics",
        "restart_mode",
        "cooling_method",
        "tinit",
        "t_final",
        "t_step",
        "time_step",
        "friction_coefficient",
        "step_num",
        "rng_seed",
        "test_rng",
        "dynamic_diag_tol",
        "potential_extrapolation",
        "initial_structure",
        "mlff_model",
        "mlff_task",
        "mlff_fmax",
        "mlff_steps",
        "mlff_optimizer",
        "mlff_device",
        "mlff_python",
        "mlff_charge",
        "mlff_spin",
        "mlff_reuse",
        "initial_density",
        "density_method",
        "ml_density_file",
        "ml_file_path",
        "ml_density_units",
        "ml_density_negative_policy",
        "ml_density_interpolation",
        "ml_density_repository",
        "ml_density_checkpoint",
        "ml_density_python",
        "ml_density_model",
        "ml_density_device",
        "ml_density_cache",
        "ml_density_regenerate",
        "ml_density_chunk_size",
        "ml_density_bridge",
        "normalize_initial_density",
        "lattice_vector_scale",
        "energy_refinement",
    }
)

#: The label of the choice (not read by Andrew Grieder's own parser: it is this merge's).
KEYWORD = "use_fortran_routines"

#: The features whose default (``Use_Fortran_Routines`` not given) is the Fortran-derived routines, as the parser
#: names them in parsec.out ("the default for ...").
SLAB_FEATURE = "slabs"
SHIFTED_MESH_FEATURE = "shifted Monkhorst-Pack grids of bulk cells"
GAMMA_FORCES_FEATURE = "forces, relaxation and molecular dynamics of bulk cells at the Gamma point"
SYMMETRIC_GRID_FEATURE = "bulk cells whose space group sets the grid"
FORTRAN_DEFAULT_FEATURES = (SLAB_FEATURE, SHIFTED_MESH_FEATURE, GAMMA_FORCES_FEATURE, SYMMETRIC_GRID_FEATURE)


@dataclass(frozen=True)
class RoutineChoice:
    """The routines of one calculation.

    ``fortran``: the Fortran-derived routines instead of Andrew Grieder's.
    ``requested``: the value of ``Use_Fortran_Routines`` in the input (``True`` or ``False``), ``None`` where the
    input does not give it (the default of each feature).  ``reasons``: what Andrew Grieder's routines do not
    compute in an input that runs on the Fortran-derived ones without asking for them (a refusal; printed with a
    warning).  ``defaults``: the features of the input whose default is the Fortran-derived routines
    (:data:`FORTRAN_DEFAULT_FEATURES`; only without the parameter).
    """

    fortran: bool = False
    requested: bool | None = None
    reasons: tuple[str, ...] = ()
    defaults: tuple[str, ...] = ()

    @property
    def automatic(self) -> bool:
        """The Fortran-derived routines run because Andrew Grieder's refuse the input (not asked for, no default)."""
        return self.fortran and self.requested is not True and bool(self.reasons) and not self.defaults

    @property
    def by_default(self) -> bool:
        """The Fortran-derived routines run as the default of a feature of the input (:data:`FORTRAN_DEFAULT_FEATURES`)."""
        return self.fortran and self.requested is None and bool(self.defaults)

    def with_reason(self, reason: str) -> "RoutineChoice":
        """The Fortran-derived routines, for ``reason`` (added to the reasons already found)."""
        reasons = self.reasons if reason in self.reasons else self.reasons + (reason,)
        return RoutineChoice(fortran=True, requested=self.requested, reasons=reasons, defaults=self.defaults)

    def with_default(self, feature: str) -> "RoutineChoice":
        """The Fortran-derived routines as the default of ``feature`` (the parameter not given)."""
        defaults = self.defaults if feature in self.defaults else self.defaults + (feature,)
        return RoutineChoice(fortran=True, requested=self.requested, reasons=self.reasons, defaults=defaults)

    def describe(self) -> str:
        """One line for parsec.out and the log."""
        if not self.fortran:
            if self.requested is False:
                return "Routines: Andrew Grieder's (Use_Fortran_Routines: .false.)"
            return "Routines: default (Andrew Grieder's); Use_Fortran_Routines: .true. selects the Fortran-derived ones"
        if self.requested is True:
            return "Routines: Fortran-derived (Use_Fortran_Routines: .true.)"
        if self.by_default:
            line = (
                "Routines: Fortran-derived, the default for " + " and ".join(self.defaults)
                + " (Use_Fortran_Routines: .false. selects Andrew Grieder's)"
            )
            if self.reasons:
                line += "; Andrew Grieder's routines would not compute: " + "; ".join(self.reasons)
            return line
        return "Routines: Fortran-derived, chosen automatically: " + "; ".join(self.reasons)

    def warning(self) -> str | None:
        """The warning of an automatic choice (a refusal), ``None`` otherwise."""
        if not self.automatic:
            return None
        return (
            "This input runs on the Fortran-derived routines (as with Use_Fortran_Routines: .true.) because "
            "Andrew Grieder's routines do not compute it: " + "; ".join(self.reasons)
        )


DEFAULT = RoutineChoice(fortran=True, requested=True)


def routines_report_line(translation) -> str | None:
    """The line of parsec.out that states the routines of a run, ``None`` where the choice changes nothing.

    It is printed for a periodic cell, for the forces and ionic loops of spinor orbitals (self-consistent spin-orbit
    coupling, non-collinear magnetism), and whenever the Fortran-derived routines run (asked for or chosen
    automatically).  Every other calculation (a cluster) runs the same code under both choices.
    """

    choice = getattr(translation, "routines", None)
    if choice is None:
        return None
    problem = translation.problem
    spinor = bool(problem.scf.self_consistent_spin_orbit or problem.scf.noncollinear)
    ionic = bool(
        translation.relaxation is not None
        or translation.molecular_dynamics is not None
        or translation.compute_forces
    )
    if choice.fortran or problem.periodic_cell is not None or (spinor and ionic):
        return " " + choice.describe()
    return None


def bulk_default_features(
    atoms,
    periodic_cell,
    grid,
    *,
    kpoints: bool,
    gamma_only: bool,
    forces: bool,
    spinors: bool,
    spin_polarized: bool,
    ignore_symmetry: bool,
    search_info: int,
    spline_species: bool = False,
) -> tuple[str, ...]:
    """The features of a bulk cell whose default is the Fortran-derived routines (besides a shifted mesh).

    ``grid`` holds the spacing and the expansion order of the input (either set of routines); ``kpoints``: the
    input samples k-points (``Monkhorst_Pack_Grid`` or ``Kpoint_List``), ``gamma_only``: that sampling is the
    Gamma point alone; ``forces``: forces are computed (a relaxation, molecular dynamics or
    ``Skip_Force: .false.``); ``spinors``: self-consistent spin-orbit coupling or non-collinear magnetism;
    ``search_info``: the ``ierr`` of the Fortran-derived parser's space-group search
    (``_bulk_symmetry_search_info``); ``spline_species``: every species uses ``Cubic_Spline``.

    * :data:`GAMMA_FORCES_FEATURE`: forces of scalar orbitals (spin-polarized or not) at the Gamma point without a
      k-point input, where the orthorhombic grid of the Fortran-derived routines has the points of Andrew
      Grieder's grid (:func:`_grids_coincide`) and every species uses ``Cubic_Spline`` (his local potential is
      always the cubic spline, theirs follows the species), so that a run with forces has the energy of the same
      run without them: in practice ``Ignore_Symmetry`` inputs with spline species.  Widening it needs the
      orthorhombic grid of the Fortran-derived routines to be Fortran's and the same local potential.  Their accelerated path is 2.5-5 times faster there (Si64 molecular dynamics on an A100: 7.6 against
      41 s; device forces, device Hartree, geometry updates 4-15 times faster) and their forces are the
      derivative of their energy.
    * :data:`SYMMETRIC_GRID_FEATURE`: a cell on Fortran's general grid whose space group's fractional
      translations change the number of points (``adjust_step``), so that Andrew Grieder's grid (no space group)
      is not Fortran's - in a run that both sets of routines give to the reference solver (k-points other than
      the Gamma point alone, spin polarization, spinors), without forces.

    Neither is chosen where the Fortran-derived routines would leave the accelerated path while Andrew
    Grieder's stay on it, or would be much slower:

    * their general grid (non-orthogonal lattice vectors, or sides without one common grid spacing) in a scalar
      unpolarized Gamma-point run: the reference CLI against his accelerated oblique path (fcc Si16 on an A100:
      418 against 15.9 s);
    * the Gamma point named as a one-point k-point sampling: the reference k-point solver against his
      accelerated Gamma path;
    * forces on their general grid: their readable local force (0.7-30 s per evaluation against 0.13-1.7 s);
    * the forces and ionic loops of spinor orbitals: his are 30-150 times faster per force evaluation.
    """

    from ..Grid.pbc import uses_general_periodic_grid

    general = uses_general_periodic_grid(periodic_cell, grid)
    features: list[str] = []
    if (
        forces
        and not spinors
        and not kpoints
        and not general
        and spline_species
        and _grids_coincide(periodic_cell, grid, ignore_symmetry)
    ):
        features.append(GAMMA_FORCES_FEATURE)
    reference_in_both = spin_polarized or spinors or (kpoints and not gamma_only)
    if (
        general
        and not forces
        and not ignore_symmetry
        and search_info != 2
        and reference_in_both
        and _space_group_sets_grid(atoms, periodic_cell, grid.spacing, search_info)
    ):
        features.append(SYMMETRIC_GRID_FEATURE)
    return tuple(features)


def _grids_coincide(periodic_cell, grid, ignore_symmetry: bool) -> bool:
    """Whether the orthorhombic grid of the Fortran-derived routines has the points of Andrew Grieder's grid.

    Theirs (``build_fortran_periodic_grid``): ``round(L_i/h)`` points per side, the step ``L_0/n_0`` along every
    axis, from the origin.  His: the largest even number of points whose step is not below ``h``
    (``fortran_points_per_axis``), the step ``L_i/n_i`` along each axis, half a step off the origin unless
    ``Ignore_Symmetry`` is set (``fortran_grid_shift``).  The same counts, the same steps to the last bit and no
    shift give the same points.
    """

    import numpy as np

    from ..Grid.pbc import fortran_grid_shift, fortran_points_per_axis

    lattice_vectors = np.asarray(periodic_cell.lattice_vectors, dtype=float)
    if any(fortran_grid_shift(lattice_vectors, ignore_symmetry)):
        return False
    sides = np.diag(lattice_vectors)
    ours = np.round(sides / grid.spacing).astype(np.int64)
    try:
        his = fortran_points_per_axis(sides, grid.spacing)
    except ValueError:
        return False
    if not np.array_equal(ours, his):
        return False
    steps = sides / ours
    return bool(np.all(steps == steps[0]))


def _space_group_sets_grid(atoms, periodic_cell, spacing: float, search_info: int) -> bool:
    """Whether the space group of the Fortran-derived routines changes the points of Fortran's general grid.

    Their parser (``_periodic_symmetry``) hands the fractional translations of the group to ``adjust_step``;
    Andrew Grieder's grid takes the largest even number of points without them (``fortran_points_per_axis``).
    """

    import numpy as np

    from ..Grid.pbc import adjust_step, fortran_points_per_axis
    from ..Grid.space_group import find_space_group
    from ..Grid.symmetry_analysis import symmetry_check

    lattice_vectors = np.asarray(periodic_cell.lattice_vectors, dtype=float)
    try:
        group = find_space_group(lattice_vectors, atoms)
    except ValueError:
        return False  # the Fortran-derived parser says why, if it runs
    if group.order <= 1 or (search_info == 0 and symmetry_check(group).error):
        return False
    lengths = np.linalg.norm(lattice_vectors, axis=1)
    try:
        his = [int(value) for value in fortran_points_per_axis(lengths, spacing)]
        ours = [
            adjust_step(float(length), spacing, translations=taus)[0]
            for length, taus in zip(lengths, group.commensurate_translations())
        ]
    except ValueError:
        return False
    return ours != his


class UseFortranRoutines(Exception):
    """Raised by the parser where an input of the default routines turns out to need the Fortran-derived ones.

    The parser starts again with :attr:`choice` (every decision of the first pass is redone).
    """

    def __init__(self, choice: RoutineChoice) -> None:
        super().__init__("; ".join(choice.reasons))
        self.choice = choice


def decide_routines(
    scalar: Mapping[str, Sequence[object]],
    blocks: Mapping[str, Sequence[object]],
    *,
    requested: bool | None = None,
) -> RoutineChoice:
    """The routines of an input, from its labels (``scalar``/``blocks`` of the parser, lower case).

    ``requested`` is the value of ``Use_Fortran_Routines`` (``None`` where the input does not give it).  ``True``
    selects the Fortran-derived routines.  Otherwise a label that Andrew Grieder's routines do not read selects them
    too (with the reason).  The parser checks the values afterwards and may add reasons, or the features whose
    default is the Fortran-derived routines (:class:`UseFortranRoutines`).
    """

    return RoutineChoice(fortran=True, requested=True)
    start = RoutineChoice(requested=requested)
    unknown = sorted(
        key for key in scalar if key != KEYWORD and key not in DEFAULT_SPECIES_KEYS | DEFAULT_GLOBAL_KEYS
    )
    unknown += [f"begin {key}" for key in sorted(blocks) if key not in DEFAULT_BLOCK_KEYS]
    if unknown:
        return start.with_reason("input option(s) " + ", ".join(unknown))
    return start


__all__ = [
    "DEFAULT",
    "FORTRAN_DEFAULT_FEATURES",
    "GAMMA_FORCES_FEATURE",
    "SHIFTED_MESH_FEATURE",
    "SLAB_FEATURE",
    "SYMMETRIC_GRID_FEATURE",
    "bulk_default_features",
    "routines_report_line",
    "DEFAULT_BLOCK_KEYS",
    "DEFAULT_GLOBAL_KEYS",
    "DEFAULT_SPECIES_KEYS",
    "KEYWORD",
    "RoutineChoice",
    "UseFortranRoutines",
    "decide_routines",
]
