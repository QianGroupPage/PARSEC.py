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
  decisions of 2 and 4 November 2026), and a weighted ``Kpoint_List`` of a bulk cell (which his routines refused
  before 8 November 2026);
* ``.false.``: Andrew Grieder's routines wherever his code accepts the input (his slab, his shifted mesh and a
  weighted list of a bulk single point too, reduced and averaged as ``Input.parsec_input._default_routines_wedge``
  and ``_default_routines_list_symmetry`` say);
* ``.true.``: the Fortran-derived routines.

An input that Andrew Grieder's routines do not compute (a keyword or a value they do not read, a geometry they
refuse) runs on the Fortran-derived routines by itself, and the reason is printed with a warning.  A keyword that
both sets read the same way (the shared eigensolvers and SCF loops, :data:`NEUTRAL_KEYS`) chooses nothing, and so
does a label written at a value that has no effect (:data:`NO_EFFECT`: label -> the predicate and its reason, such as
``Add_Point_Charges: .false.`` or ``Mixing_Restart_Factor 0.3``, the lines PARSEC 1.5's bak_complete_parsec.in writes).
:func:`decide_routines` makes the choice once per run, from the input's keywords; the parser adds the reasons it
finds in the values and the features whose default is the Fortran-derived routines
(``parsec_python.Input.parsec_input``) and records the result on the translation
(``ParsecInputTranslation.routines``) and on the periodic grid settings (``PeriodicGridSettings.fortran_routines``),
which every periodic routine reads.  Both CLIs follow it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .solver_options import SOLVER_KEYS

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
        # PARSEC 1.5's grid shift: his grid reads any shift (the parser sets it; his slab, which has none, falls
        # back by name)
        "grid_shift",
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
        # PARSEC 1.5's blocks of the domain: a cluster's domain (the same code under both choices) or, for a bulk
        # cell, the grid of 1 with the symmetry switched off (the parser refuses what neither computes)
        "foldblock_side_length",
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

#: Labels of PARSEC 1.5 that the merged parser reads and whose effect is the same under both sets of routines, so
#: that they decide nothing (the parity round of 2026-11-04, input layer): keywords that only tune 1.5's MPI
#: layout, ScaLAPACK, ELPA, HDF5, space-filling curves, spectrum slicing, threads or its internal tests (no effect
#: here); ``Use_LibXC``/``Func_Id_*``/``LibXC_Functional``/``LibXC_Enforce_FHC`` (libxc's functionals: the XC
#: evaluation both sets share, ``V_xc.selection``);
#: ``Pseudopot_File`` (the parser resolves the file for both); the Poisson CG controls (``HartreeSettings``, read by
#: the solvers both sets share); the output switches ``Generate_Complete_Parsec_In``, ``Output_Ascii_Data`` and
#: ``Print_Unconverged_Density``.  The renamed ``Mol_RNG_Seed`` is read as ``RNG_Seed`` before the choice.
#: Since 2026-11-04 (evening) also the eigensolver and SCF
#: controls of the earlier parity round that the shared eigensolvers and SCF loops read the same way under both
#: sets (``Convergence_Criterion_Approach``, ``Chebyshev_Filtering``, ``Maximum_Matvec``,
#: ``Fix_Number_Eigenvalues``) and the MPI tuning labels that act on nothing here (``MPI_Groups``, the earlier name
#: of ``Npcol``, ``Exp_Comm``, ``Max_Lap_Buffers``).
NEUTRAL_KEYS = frozenset(
    {
        "use_scalapack",
        "scalapack_block_size",
        "use_elpa",
        "elpa_block_size",
        "npcol",
        "spectrum_slicing",
        "use_krylov_subspace_method",
        "use_lanczos_dos_estimation",
        "spectrum_slicing_cycles_early_quit",
        "spectrum_slicing_max_cycles",
        "spectrum_slicing_max_cycles_first",
        "num_of_iter_still_using_chebyshev",
        "polym_first_ss_iter",
        "polym_ss",
        "num_of_copied_groups",
        "num_of_slice1_filtering",
        "nslices",
        "spectrum_slicing_ratio",
        "use_sfc",
        "thread_random_init",
        "rritz_logic",
        "test_logic",
        "max_taskgroups",
        "vxc_matrix_elements",
        "use_hdf5",
        "use_libxc",
        "func_id_x",
        "func_id_c",
        "libxc_functional",
        "libxc_enforce_fhc",
        "libxc_density_threshold",
        "libxc_tau_threshold",
        "pseudopot_file",
        "hartree_cg_tolerance",
        "hartree_cg_criterion",
        "generate_complete_parsec_in",
        "output_ascii_data",
        "print_unconverged_density",
        # Eigensolver and SCF controls of the Fortran parity round: shared code under both sets.
        "convergence_criterion_approach",
        "chebyshev_filtering",
        "maximum_matvec",
        "fix_number_eigenvalues",
        # MPI tuning labels of the earlier PARSEC: read, warned about, no effect.
        "mpi_groups",
        "exp_comm",
        "max_lap_buffers",
        # FDET of PARSEC 1.5 and its PARSEC.py switches (FDET_KEYWORDS of parsec_input): FDET is
        # for clusters only, and a periodic input that asks for it is refused by name (_fdet_settings); the
        # values that ask for nothing (Potential_Field: .false., which 1.5's bak_complete_parsec.in writes in
        # every run, Write_Potential_Field: .false., AFM_Scan: .false., or a functional without Potential_Field)
        # are no-ops under both sets (since 2026-11-05).
        "potential_field",
        "potential_field_name",
        "kinetic_energy_functional",
        "pbe_m",
        "pbe_k",
        "boundary_sphere_around_tip",
        "fdet_kinetic_form",
        "write_potential_field",
        "afm_scan",
        "afm_scan_heights",
        "afm_scan_height_step",
        "afm_scan_line_range",
        # The keywords of a self-consistent hybrid functional (parsec_input._HYBRID_KEYS): the Fock loop of
        # SCF/hybrid.py and SCF/hybrid_periodic.py reads them the same way under both sets (since 2026-11-07,
        # when periodic cells took hybrids).
        "exact_exchange_fraction",
        "screening_parameter",
        "exx_max_iter",
        "exx_tolerance",
        "exx_batch_size",
        "exx_pair_solver",
        "exx_inner_criterion",
        "exx_mixer_history",
        "exx_outer_mixing",
        "exx_divergence",
        "exx_slab_coulomb",
        "exx_auxiliary_exponent",
        # The Kerker preconditioner of the mixing of bulk cells (PARSEC.py, Mixer/preconditioner.py, 2026-11-08):
        # the shared SCF loops wrap their mixer the same way under both sets.
        "mixing_preconditioner",
        "kerker_wavevector",
        "kerker_dielectric_constant",
    }
    # The eigensolver, SCF-control and start-vector labels of PARSEC 1.5 and the
    # extensions of its round (parsec_python.Input.solver_options): options of
    # the eigensolvers and SCF loops that both sets of routines share.
    | SOLVER_KEYS
)

#: The label of the choice (not read by Andrew Grieder's own parser: it is this merge's).
KEYWORD = "use_fortran_routines"

#: The features whose default (``Use_Fortran_Routines`` not given) is the Fortran-derived routines, as the parser
#: names them in parsec.out ("the default for ...").
SLAB_FEATURE = "slabs"
SHIFTED_MESH_FEATURE = "shifted Monkhorst-Pack grids of bulk cells"
GAMMA_FORCES_FEATURE = "forces, relaxation and molecular dynamics of bulk cells at the Gamma point"
SYMMETRIC_GRID_FEATURE = "bulk cells whose space group sets the grid"
#: A weighted Kpoint_List of a bulk cell: Andrew Grieder's routines refused it before 2026-11-08 (it ran on the
#: Fortran-derived routines); they run it now with Use_Fortran_Routines: .false., the default keeps those routines.
WEIGHTED_LIST_FEATURE = "Kpoint_Lists of unequal weights of bulk cells"
FORTRAN_DEFAULT_FEATURES = (
    SLAB_FEATURE, SHIFTED_MESH_FEATURE, GAMMA_FORCES_FEATURE, SYMMETRIC_GRID_FEATURE, WEIGHTED_LIST_FEATURE
)

#: Labels of PARSEC 1.5 for the forces, ionic loops and band structure (the parity round of 2026-11-04)
#: whose effect is the same under both sets of routines, so that they decide nothing: ``Calculate_Force``
#: (read as ``Skip_Force``, which both read), ``Use_Old_Forces`` (no effect), ``Write_Relax_Restart`` (the file of
#: the shared relaxation recorder) and ``Bands_FF_MaxIter`` (the shared band-structure solve).
#: ``Recenter_Atoms_In_Aperiodic_Directions`` is one of them since 2026-11-05: both
#: values act the same under both sets.  ``.true.`` is the default.  ``.false.`` keeps the coordinates of the input,
#: which Andrew Grieder's routines do anyway for a bulk cell (the start-up ``back_to_cell`` of
#: ``parsec_input`` runs for the Fortran-derived routines only); the re-centring of a slab along z (the parser
#: and ``Relaxation/run.py``) and the frame of a molecular-dynamics restart (``MolecularDynamics/run.py``) are
#: shared code that reads the value for both; wires are not re-centred.
SMALL15_SHARED_KEYS = frozenset(
    {
        "calculate_force",
        "use_old_forces",
        "write_relax_restart",
        "bands_ff_maxiter",
        "recenter_atoms_in_aperiodic_directions",
    }
)

#: Blocks that choose nothing for the same reason: the lateral points of an FDET AFM scan (``FDET_BLOCKS``).
NEUTRAL_BLOCK_KEYS = frozenset({"afm_scan_lateral_points"})


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
    requested_shift=None,
) -> tuple[str, ...]:
    """The features of a bulk cell whose default is the Fortran-derived routines (besides a shifted mesh).

    ``grid`` holds the spacing and the expansion order of the input (either set of routines); ``kpoints``: the
    input samples k-points (``Monkhorst_Pack_Grid`` or ``Kpoint_List``), ``gamma_only``: that sampling is the
    Gamma point alone; ``forces``: forces are computed (a relaxation, molecular dynamics or
    ``Skip_Force: .false.``); ``spinors``: self-consistent spin-orbit coupling or non-collinear magnetism;
    ``search_info``: the ``ierr`` of the Fortran-derived parser's space-group search
    (``_bulk_symmetry_search_info``); ``spline_species``: every species uses ``Cubic_Spline``;
    ``requested_shift``: the ``Grid_Shift`` block of PARSEC 1.5 (``None`` without it), which both grids take.

    * :data:`GAMMA_FORCES_FEATURE`: forces of scalar orbitals (spin-polarized or not) at the Gamma point without a
      k-point input, where the orthorhombic grid of the Fortran-derived routines has the points of Andrew
      Grieder's grid (:func:`_grids_coincide`) and every species uses ``Cubic_Spline`` (his local potential is
      always the cubic spline, theirs follows the species), so that a run with forces has the energy of the same
      run without them.  Since the parity round of PARSEC 1.5 their orthorhombic grid is Fortran's (``adjust_step``
      with the space group's translations, half a step off the origin unless ``Ignore_Symmetry``), so the grids
      coincide for symmetric cells too (a cube of a symmorphic group, or one whose translations
      ``adjust_step`` meets with the even count anyway, as the 8-atom diamond cube at the Gamma point, whose
      operations fail ``symchk``), not only for ``Ignore_Symmetry`` inputs.  Their accelerated path is 2.5-5 times faster there (Si64 molecular dynamics on an A100: 7.6 against
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

    from dataclasses import replace

    from ..Grid.pbc import uses_general_periodic_grid

    features: list[str] = []
    gamma_forces = forces and not spinors and not kpoints and spline_species
    reference_in_both = spin_polarized or spinors or (kpoints and not gamma_only)
    symmetric_grid = not forces and not ignore_symmetry and search_info != 2 and reference_in_both
    if not (gamma_forces or symmetric_grid):
        return ()
    # The grid settings of the Fortran-derived routines: the space group's fractional translations (as their
    # parser finds them; a search only where a feature may follow), and their shift.
    ours = replace(
        grid,
        fractional_translations=_fortran_translations(atoms, periodic_cell, ignore_symmetry, search_info),
        shift=_fortran_shift(periodic_cell, ignore_symmetry, requested_shift),
        points_rule="round",
        fortran_routines=True,
    )
    general = uses_general_periodic_grid(periodic_cell, ours)
    if (
        gamma_forces
        and not general
        and _grids_coincide(periodic_cell, ours, ignore_symmetry, requested_shift)
    ):
        features.append(GAMMA_FORCES_FEATURE)
    if (
        symmetric_grid
        and general
        and _space_group_sets_grid(atoms, periodic_cell, grid.spacing, search_info)
    ):
        features.append(SYMMETRIC_GRID_FEATURE)
    return tuple(features)


def _fortran_translations(atoms, periodic_cell, ignore_symmetry: bool, search_info: int):
    """The fractional translations the Fortran-derived parser hands to ``adjust_step`` (``_periodic_symmetry``):
    none with ``Ignore_Symmetry`` or without a search (``search_info`` 2), none where ``symchk`` fails (``search_info``
    0), else those of the space group of the atoms."""

    import numpy as np

    from ..Grid.space_group import find_space_group
    from ..Grid.symmetry_analysis import symmetry_check

    if ignore_symmetry or search_info == 2:
        return ()
    try:
        group = find_space_group(np.asarray(periodic_cell.lattice_vectors, dtype=float), atoms)
    except ValueError:
        return ()  # the Fortran-derived parser says why, if it runs
    if group.order <= 1 or (search_info == 0 and symmetry_check(group).error):
        return ()
    return group.commensurate_translations()


def _fortran_shift(periodic_cell, ignore_symmetry: bool, requested_shift):
    from ..Grid.pbc import periodic_grid_shift

    return periodic_grid_shift(
        periodic_cell.lattice_vectors, ignore_symmetry=ignore_symmetry, requested=requested_shift
    )


def _andrew_shift(periodic_cell, ignore_symmetry: bool, requested_shift):
    """The shift of Andrew Grieder's grid as the parser sets it (``fortran_grid_shift``, or the ``Grid_Shift``
    block)."""

    from ..Grid.pbc import fortran_grid_shift

    if requested_shift is None:
        return fortran_grid_shift(periodic_cell.lattice_vectors, ignore_symmetry)
    return _fortran_shift(periodic_cell, ignore_symmetry, requested_shift)


def _grids_coincide(periodic_cell, grid, ignore_symmetry: bool, requested_shift=None) -> bool:
    """Whether the orthorhombic grid of the Fortran-derived routines has the points of Andrew Grieder's grid.

    Theirs (``build_fortran_periodic_grid``, ``grid`` their settings): ``adjust_step``'s counts with the space
    group's translations (:func:`parsec_python.Grid.pbc.fortran_orthorhombic_counts`), the step ``L_0/n_0`` along
    every axis, the shift ``grid.shift``.  His: the largest even number of points whose step is not below ``h``
    (``fortran_points_per_axis``, without the translations and without PARSEC 1.5's slack ``1e-15``), the step
    ``L_i/n_i`` along each axis, half a step off the origin unless ``Ignore_Symmetry`` is set
    (``fortran_grid_shift``) or the ``Grid_Shift`` block.  The same counts, the same steps to the last bit and the
    same shift give the same points.
    """

    import numpy as np

    from ..Grid.pbc import fortran_orthorhombic_counts, fortran_points_per_axis

    lattice_vectors = np.asarray(periodic_cell.lattice_vectors, dtype=float)
    if tuple(grid.shift) != tuple(_andrew_shift(periodic_cell, ignore_symmetry, requested_shift)):
        return False
    ours = fortran_orthorhombic_counts(periodic_cell, grid)
    if ours is None:
        return False
    sides = np.diag(lattice_vectors)
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

    from ..Grid.pbc import adjusted_steps, fortran_points_per_axis
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
        ours = adjusted_steps(lengths, spacing, group.commensurate_translations())[0]
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


# ---------------------------------------------------------------------------------------------------------------
# Labels that Andrew Grieder's routines do not read, at values that do nothing (since 2026-11-05: each such value
# gives the identical translation).  PARSEC 1.5's bak_complete_parsec.in writes most of them in every run, at
# their defaults; written so they must not move a periodic input to the Fortran-derived routines.  A value that
# acts keeps choosing them (and the parser's refusals and fall-backs keep applying).
# ---------------------------------------------------------------------------------------------------------------

_LOGICAL_TRUE = {"yes", "true", "t", "1", "on"}
_LOGICAL_FALSE = {"no", "false", "f", "0", "off"}


def _texts(scalar: Mapping[str, Sequence[object]], key: str) -> list[str]:
    """The written values of ``key`` (one per occurrence: species labels have one per species)."""

    return [str(getattr(item, "value", "")) for item in scalar.get(key, ())]


def _logical(text: str) -> bool | None:
    """An ESDF logical as the parser reads it (``_boolean``), ``None`` for anything else."""

    normalized = "".join(char for char in text.lower() if char not in "._- \t")
    if normalized in _LOGICAL_TRUE:
        return True
    if normalized in _LOGICAL_FALSE:
        return False
    return None


def _number(text: str) -> float | None:
    """The first number of a value (a unit may follow; Fortran's ``d`` exponent read), ``None`` otherwise."""

    words = text.split()
    if not words:
        return None
    try:
        return float(words[0].lower().replace("d", "e"))
    except ValueError:
        return None


def _is_true(key: str, scalar) -> bool:
    """``key`` is written and every occurrence is a true logical."""

    texts = _texts(scalar, key)
    return bool(texts) and all(_logical(text) is True for text in texts)


def _false(key: str):
    return lambda scalar, blocks: all(_logical(text) is False for text in _texts(scalar, key))


def _equals(key: str, value: float):
    return lambda scalar, blocks: all(_number(text) == value for text in _texts(scalar, key))


def _word(key: str, *words: str):
    return lambda scalar, blocks: all(text.strip().strip("'\"").lower() in words for text in _texts(scalar, key))


def _either(*tests):
    return lambda scalar, blocks: any(test(scalar, blocks) for test in tests)


def _unless_true(other: str):
    """No effect unless the logical ``other`` is true (the label is read only then)."""

    return lambda scalar, blocks: not _is_true(other, scalar)


def _always(scalar, blocks) -> bool:
    return True


def _no_nscf_run(scalar, blocks) -> bool:
    """No non-self-consistent run (``nscf_kpoints``/``nscf_kgrid`` block) and no BerkeleyGW output."""

    return not any(key in blocks for key in ("nscf_kpoints", "nscf_kgrid", "nscf_kgrid_shift")) and not _is_true(
        "output_gw", scalar
    )


def _with_kpoints(scalar, blocks) -> bool:
    """A k-point run (``Kpoint_Method`` mp or manual with its grid or list): eigen.dat is written whatever
    ``Output_Eigenvalues`` says (the parser's ``eigenvalues = Output_Eigenvalues or has_kpoints``)."""

    methods = [text.strip().lower() for text in _texts(scalar, "kpoint_method")]
    if methods == ["mp"]:
        return "monkhorst_pack_grid" in blocks
    if methods == ["manual"]:
        return "kpoint_list" in blocks or bool(_texts(scalar, "kpoint_file"))
    return False


def _no_core_filter(scalar, blocks) -> bool:
    """``Core_Filter 0`` with every ``Alpha_Filter`` and ``Beta1_Filter`` 0: no Fourier filter at all (the parser
    then sets none; ``SpeciesPotential.has_fourier_filter``)."""

    return all(
        all(_number(text) == 0.0 for text in _texts(scalar, key))
        for key in ("core_filter", "alpha_filter", "beta1_filter")
    )


@dataclass(frozen=True)
class NoEffect:
    """When a label, as written, does nothing: ``test(scalar, blocks)`` and why (``reason``)."""

    test: object
    reason: str


#: Label -> the predicate "has no effect" and its reason.  Every entry is checked by
#: ``tests/unit/test_routines.py`` (``ValueAwareRoutineTests``): the no-op values give the translation of the input
#: without the line, on the default routines; the acting values still choose the Fortran-derived ones.
NO_EFFECT: dict[str, NoEffect] = {
    "add_point_charges": NoEffect(_false("add_point_charges"), "false: no point charges (the default)"),
    "point_typ_num": NoEffect(
        _either(_equals("point_typ_num", 0), _unless_true("add_point_charges")),
        "0, or without Add_Point_Charges: no point charges",
    ),
    "add_charged_sheets": NoEffect(_false("add_charged_sheets"), "false: no charged sheets (the default)"),
    "num_charged_sheets": NoEffect(
        _either(_equals("num_charged_sheets", 0), _unless_true("add_charged_sheets")),
        "0, or without Add_Charged_Sheets: no charged sheets",
    ),
    "alpha_filter": NoEffect(_equals("alpha_filter", 0.0), "0 (the default): no Fourier filter"),
    "beta1_filter": NoEffect(_equals("beta1_filter", 0.0), "0 (the default): no Fourier filter"),
    "core_filter": NoEffect(_no_core_filter, "0 with Alpha_Filter and Beta1_Filter 0: no Fourier filter"),
    "calc_hartree_g": NoEffect(_false("calc_hartree_g"), "false (the default): the real-space Hartree solver"),
    "create_dos": NoEffect(_false("create_dos"), "false (the default): no DOS"),
    "dos_pnum": NoEffect(
        _either(_equals("dos_pnum", 1000), _unless_true("create_dos")),
        "1000 (the default), or without Create_DOS (read for the DOS only)",
    ),
    "ylm_dos_l": NoEffect(
        _either(_equals("ylm_dos_l", -1), _unless_true("create_dos")),
        "-1 (the default), or without Create_DOS (read for the DOS only)",
    ),
    "external_mag_field": NoEffect(
        _equals("external_mag_field", 0.0), "0 (the default): no field and real wavefunctions"
    ),
    "hf_one": NoEffect(
        _either(_equals("hf_one", 0), _unless_true("van_der_waals")), "0, or without Van_Der_Waals (ignored)"
    ),
    "hf_two": NoEffect(
        _either(_equals("hf_two", 0), _unless_true("van_der_waals")), "0, or without Van_Der_Waals (ignored)"
    ),
    "hf_three": NoEffect(
        _either(_equals("hf_three", 0), _unless_true("van_der_waals")), "0, or without Van_Der_Waals (ignored)"
    ),
    "van_der_waals": NoEffect(_false("van_der_waals"), "false (the default): no TS correction"),
    "ldaplusu_u": NoEffect(_equals("ldaplusu_u", 0.0), "0 for every species: no LDA+U term"),
    "ldaplusu_j": NoEffect(_equals("ldaplusu_j", 0.0), "0 for every species: no LDA+U term"),
    "mixing_en_like": NoEffect(_equals("mixing_en_like", 0), "0, the default of the mixer"),
    "mixing_group_size": NoEffect(_equals("mixing_group_size", 1), "1, the default of the mixer"),
    "mixing_memory_expand_factor": NoEffect(
        _equals("mixing_memory_expand_factor", 2.0), "2, the default of the mixer"
    ),
    "mixing_preferred_type": NoEffect(_equals("mixing_preferred_type", 1), "1, the default of the mixer"),
    "mixing_restart_factor": NoEffect(_equals("mixing_restart_factor", 0.3), "0.3, the default of the mixer"),
    "nscf_fermi_level": NoEffect(_no_nscf_run, "without a non-self-consistent run or Output_GW (not read)"),
    "nscf_kpoint_unit": NoEffect(_no_nscf_run, "without a non-self-consistent run or Output_GW (not read)"),
    "nscf_states_num": NoEffect(_no_nscf_run, "without a non-self-consistent run or Output_GW (not read)"),
    "orb_mag": NoEffect(_always, "any value: accepted as in PARSEC, where it changes nothing (a warning)"),
    "output_gw": NoEffect(_false("output_gw"), "false (the default): no BerkeleyGW files"),
    "polarizability": NoEffect(_false("polarizability"), "false (the default): no polarizability run"),
    "restart_from_wfndat": NoEffect(
        _either(_false("restart_from_wfndat"), _unless_true("restart_run")),
        "false, or without Restart_Run (read and unused, parsec.F90 l.917-933)",
    ),
    "symmetrize_charge_density": NoEffect(
        _false("symmetrize_charge_density"), "false (the default): the density is not symmetrized"
    ),
    # Output switches at their defaults: the same files and the same parsec.out.
    "output_data": NoEffect(lambda scalar, blocks: all(_logical(t) is True for t in _texts(scalar, "output_data")),
                            "true (the default): parsec.dat is written"),
    "output_eigenvalues": NoEffect(
        _either(_false("output_eigenvalues"), _with_kpoints),
        "false (the default; the notice 'No eigenvalue output File!' PARSEC prints for the written label comes "
        "from the output both sets share), or any value with k-points, which write eigen.dat anyway",
    ),
    "save_intermediate_eigenvalues": NoEffect(_false("save_intermediate_eigenvalues"), "false (the default)"),
    "save_intermediate_charge_density": NoEffect(_false("save_intermediate_charge_density"), "false (the default)"),
    "output_file_name": NoEffect(_word("output_file_name", "parsec.out"), "parsec.out (the default)"),
    "output_grid_data": NoEffect(_word("output_grid_data", "none", ""), "none (the default): no grid files"),
}


def _bulk_kpoint_reduction(scalar, blocks) -> bool:
    """A bulk cell with a ``Monkhorst_Pack_Shift`` mesh or a ``Kpoint_List``/``Kpoint_File``: the inputs whose density
    the default routines average (``Input.parsec_input._default_routines_wedge`` and ``_default_routines_list_symmetry``,
    2026-11-08).  The parser sends one it does not average (forces, a slab, ...) to the Fortran-derived routines."""

    conditions = [text.strip().lower() for text in _texts(scalar, "boundary_conditions")]
    if conditions != ["bulk"]:
        return False
    methods = [text.strip().lower() for text in _texts(scalar, "kpoint_method")]
    if methods == ["mp"]:
        return "monkhorst_pack_shift" in blocks
    if methods == ["manual"]:
        return "kpoint_list" in blocks or bool(_texts(scalar, "kpoint_file"))
    return False


#: Labels the default routines read for some inputs only (label -> the predicate of those inputs): elsewhere they
#: choose the Fortran-derived routines as every label of neither set does.
DEFAULT_CONDITIONAL_KEYS = {"symmetrize_charge_density": _bulk_kpoint_reduction}


def has_no_effect(key: str, scalar: Mapping[str, Sequence[object]], blocks: Mapping[str, Sequence[object]]) -> bool:
    """Whether ``key`` is written at a value that does nothing (:data:`NO_EFFECT`); ``False`` for other labels."""

    entry = NO_EFFECT.get(key)
    if entry is None or not _texts(scalar, key):
        return False
    try:
        return bool(entry.test(scalar, blocks))
    except (TypeError, ValueError):
        return False


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
        key
        for key in scalar
        if key != KEYWORD
        and key not in DEFAULT_SPECIES_KEYS | DEFAULT_GLOBAL_KEYS | NEUTRAL_KEYS | SMALL15_SHARED_KEYS
        # a label written at a value that has no effect chooses nothing (NO_EFFECT)
        and not has_no_effect(key, scalar, blocks)
        # nor one the default routines read for this input (DEFAULT_CONDITIONAL_KEYS)
        and not (key in DEFAULT_CONDITIONAL_KEYS and DEFAULT_CONDITIONAL_KEYS[key](scalar, blocks))
    )
    # the values of the labels of NO_EFFECT that act, so that the warning names them
    unknown = [
        f"{key} = {' / '.join(_texts(scalar, key))}" if key in NO_EFFECT else key for key in unknown
    ]
    unknown += [
        f"begin {key}" for key in sorted(blocks) if key not in DEFAULT_BLOCK_KEYS | NEUTRAL_BLOCK_KEYS
    ]
    if unknown:
        return start.with_reason(
            "input option(s) " + ", ".join(unknown) + " (read by the Fortran-derived routines only)"
        )
    return start


__all__ = [
    "DEFAULT",
    "FORTRAN_DEFAULT_FEATURES",
    "GAMMA_FORCES_FEATURE",
    "SHIFTED_MESH_FEATURE",
    "SLAB_FEATURE",
    "SMALL15_SHARED_KEYS",
    "SYMMETRIC_GRID_FEATURE",
    "WEIGHTED_LIST_FEATURE",
    "bulk_default_features",
    "routines_report_line",
    "DEFAULT_BLOCK_KEYS",
    "DEFAULT_CONDITIONAL_KEYS",
    "DEFAULT_GLOBAL_KEYS",
    "DEFAULT_SPECIES_KEYS",
    "KEYWORD",
    "NEUTRAL_BLOCK_KEYS",
    "NEUTRAL_KEYS",
    "NO_EFFECT",
    "NoEffect",
    "has_no_effect",
    "RoutineChoice",
    "UseFortranRoutines",
    "decide_routines",
]
