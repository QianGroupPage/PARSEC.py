"""The effective input of a run, as a ``parsec.in``: PARSEC 1.5's ``bak_complete_parsec.in``.

PARSEC 1.5 writes ``bak_complete_parsec.in`` when the input says ``Generate_Complete_Parsec_In: .true.``
(``src/io/usrinputfile.F90:4968-5000`` and ``write_out_parameters``, 5477-6042): every parameter of the run,
"in atomic units at this stage", with the lattice vectors and the atoms already expanded by ``Super_Cell`` (which is
then 1).  :func:`complete_input_text` writes the same for this port, from the translation of the input: the
values the run uses, defaults included, lengths in bohr, energies in Ry, temperatures in K, charges in e, the
cell and the atoms after ``Super_Cell`` and ``Super_Cell_Vac`` (coordinates in Cartesian bohr, the movable atoms
in a fourth column), each species with its pseudopotential file (paths relative to the input).  Keywords whose
value the translation does not hold in one place (k-points, band paths, output files, molecular dynamics, ...) are
copied from the input as they are, in its order, but for the k-point lists of an expanded ``Super_Cell``, which are
written in 1/bohr.  The file is an input of this port: read again it gives the same problem, unless a path holds a
comment character of ESDF (``#``, ``;``, ``!``; the header then says so).

1.5's own file has defects this one does not copy: ``full_hartree`` printed from an uninitialised local variable,
``core_cutoff_radius``, ``alpha_filter``, ``beta1_filter``, ``core_filter`` and ``nonlinear_core_correction`` of
species and boundary conditions that never read them (uninitialised, e.g. ``6.68e-310`` and ``T``),
``cluster_domain_shape undef`` and empty blocks.  Keywords of 1.5 that this port
does not read are not written.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

#: The file name of PARSEC 1.5 (in the directory of the input).
COMPLETE_INPUT_NAME = "bak_complete_parsec.in"

_LOCAL_LABELS = ("s", "p", "d", "f")

#: Keywords the writer reconstructs from the translation (any other keyword of the input is copied as it is).
_RECONSTRUCTED = frozenset(
    {
        "boundary_conditions",
        "boundary_sphere_radius",
        "cluster_domain_shape",
        "domain_shape_parameters",
        "grid_spacing",
        "expansion_order",
        "ignore_symmetry",
        "double_grid_order",
        "full_hartree",
        "cell_shape",
        "lattice_vector_scale",
        "super_cell",
        "super_cell_vac",
        "coordinate_unit",
        "eigensolver",
        "diag_tolerance",
        "chebdav_degree",
        "ff_maxiter",
        "matvec_blocksize",
        "chebyshev_degree",
        "chebyshev_degree_delta",
        "subspace_buffer_size",
        "chebyshev_filtering",
        "maximum_matvec",
        "dynamic_diag_tol",
        # PARSEC 1.5 writes the effective (randomised) seed; FF_Residual_Test
        # is dropped where Diag_Tolerance <= 0 switches it off.
        "vector_rng_seed",
        "ff_residual_test",
        "exact_state_count",
        "mixing_method",
        "mixing_param",
        "memory_param",
        "restart_mixing",
        "mixing_group_size",
        "mixing_en_like",
        "mixing_restart_factor",
        "mixing_memory_expand_factor",
        "mixing_preferred_type",
        "mixing_safeguard",
        "mixing_step_limit",
        "mixing_growth_trigger",
        "mixing_backoff",
        "solver_lpole",
        "hartree_boundary_tolerance",
        "hartree_atomic_tail",
        "domain_energy_tolerance",
        "hartree_cg_tolerance",
        "hartree_cg_criterion",
        "max_iter",
        "convergence_criterion",
        "convergence_criterion_approach",
        "use_plain_sre",
        "states_num",
        "net_charges",
        "fermi_temp",
        "spin_polarization",
        "correlation_type",
        "ion_energy_diff",
        # the Fock loop of a hybrid Correlation_Type (parsec_input._HYBRID_KEYS)
        "exact_exchange_fraction",
        "screening_parameter",
        "exx_max_iter",
        "exx_tolerance",
        "exx_batch_size",
        "exx_pair_solver",
        "exx_inner_criterion",
        "exx_mixer_history",
        "exx_outer_mixing",
        "spin_symmetry_sectors",
        "atom_types_num",
        "total_atom_num",
        "add_point_charges",
        "point_typ_num",
        "pt_chg",
        "point_coord",
        "add_charged_sheets",
        "num_charged_sheets",
        "sheet_charge",
        "sheet_coord",
        "minimization",
        "movement_num",
        "force_min",
        "max_step",
        "min_step",
        "bfgs_number_corr",
        "relax_restart",
        "relaxation_scf_upscale",
        "output_level",
        "output_all_states",
        "generate_complete_parsec_in",
        "use_fortran_routines",
        # the species (one section each, in the order of the input)
        "atom_type",
        "element_symbol",
        "pseudopotential_format",
        "pseudopot_file",
        "local_component",
        "potential_num",
        "core_cutoff_radius",
        "electron_per_orbital",
        "nonlinear_core_correction",
        "so_psp",
        "read_vcd",
        "cubic_spline",
        "initial_spin_polarization",
        "atomic_radius",
        "ldaplusu_u",
        "ldaplusu_j",
        "move_flag",
        "alpha_filter",
        "beta1_filter",
        "core_filter",
        "atomic_energy_correction",
        "ylm_dos_cutoff",
        "atom_coord",
        "initial_ncl_moment",
        # FDET of PARSEC 1.5 and the PARSEC.py switches of FDET (_fdet_lines)
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
        "afm_scan_lateral_points",
    }
)


#: ESDF's comment characters (esdf_mod.f90 ``comment``): everything from the first of them to the end of a line
#: is dropped, quotes or not, by PARSEC 1.5 and by this reader.
_COMMENT_CHARACTERS = ("#", ";", "!")


def _path_text(path, directory, label: str, unreadable: list[str]) -> str:
    """``path`` as written in the bak file: relative to ``directory`` (the input's, where the file is written).

    The readers look for a relative ``Pseudopot_File`` and ``Potential_Field_Name`` beside the input first, so the
    relative path reads back to the same file; it is also free of the comment characters of the folders above.
    A file on another drive keeps its absolute path.  A path that still holds a comment character cannot be read
    back by ESDF (it has no quoting); ``unreadable`` collects the label for the warning of the header.
    """

    import os

    try:
        text = os.path.relpath(os.fspath(path), os.fspath(directory))
    except ValueError:  # another drive (Windows)
        text = os.fspath(path)
    if any(character in text for character in _COMMENT_CHARACTERS):
        unreadable.append(f"{label} {text}")
    return text


def _fdet_lines(translation, unreadable: list[str]) -> list[str]:
    """FDET (``Potential_Field``), ``Write_Potential_Field`` and ``AFM_Scan`` as the run uses them.

    The sample file is written relative to the input (:func:`_path_text`); nothing is written for a run without them.
    """

    lines: list[str] = []
    settings = getattr(translation.problem, "embedding", None)
    if settings is not None:
        directory = translation.source.parent
        lines += [
            "",
            "# Frozen-density embedding (PARSEC 1.5)",
            "Potential_Field  .true.",
            "Potential_Field_Name  "
            + _path_text(settings.field_path, directory, "Potential_Field_Name", unreadable),
            f"Kinetic_Energy_Functional  {'no' if settings.kinetic_functional == 'none' else settings.kinetic_functional}",
        ]
        if settings.kinetic_functional == "pb":
            lines += [f"pbe_m  {_number(settings.pbe_mu)}", f"pbe_k  {_number(settings.pbe_kappa)}"]
        lines += [
            f"FDET_Kinetic_Form  {settings.kinetic_form}",
            f"Boundary_Sphere_Around_Tip  {_logical(settings.tip_sphere)}",
        ]
    scan = getattr(translation, "afm_scan", None)
    if scan is not None:
        lines += [
            "AFM_Scan  .true.",
            f"AFM_Scan_Heights  {int(scan.heights)}",
            f"AFM_Scan_Height_Step  {_number(scan.height_step)} bohr",
            "begin AFM_Scan_Lateral_Points",
            *(_row(point) for point in scan.lateral_points),
            "end AFM_Scan_Lateral_Points",
        ]
        if scan.line_range is not None:
            lines.append(f"AFM_Scan_Line_Range  {scan.line_range[0]} {scan.line_range[1]}")
    if getattr(translation, "write_potential_field", False):
        lines.append("Write_Potential_Field  .true.")
    return lines


def _number(value: float) -> str:
    """A float that reads back to the same bits (``repr``)."""

    return repr(float(value))


def _logical(value: bool) -> str:
    return ".true." if value else ".false."


def _row(values: Sequence[float]) -> str:
    return "  " + "  ".join(_number(value) for value in values)


def _boundary(translation) -> str:
    problem = translation.problem
    if problem.periodic_cell is None:
        return "cluster"
    if getattr(problem.periodic_cell, "dimensions", 3) == 2:
        return "slab"  # Andrew Grieder's slab
    periodicity = getattr(problem.grid, "periodicity", 3)
    return {1: "wire", 2: "slab", 3: "bulk"}[periodicity]


def complete_input_text(
    translation,
    scalar: Mapping[str, Sequence[object]],
    blocks: Mapping[str, Sequence[object]],
    *,
    kpoint_cell=None,
) -> str:
    """The text of ``bak_complete_parsec.in`` for ``translation``.

    ``scalar`` and ``blocks`` are the labels of the input as the parser read them (lower case, ``_InputItem``
    lists): the keywords that are not reconstructed are copied from them, and they say which optional keywords the
    input gives.  ``kpoint_cell`` is the cell with which the reader converted the k-point lists of the input in
    ``reciprocal_lattice_vectors`` (the input cell, before ``Super_Cell``; :func:`_kpoint_list_lines`).

    ``Periodic_System`` is copied as the input gives it: the reader infers it from ``Boundary_Conditions``, but the
    routine decision of a slab reads the line itself (``Input.parsec_input``).  Paths are written relative to the
    input (:func:`_path_text`).
    """

    from ..V_xc.selection import FORTRAN_CORRELATION_CODES

    problem = translation.problem
    grid = problem.grid
    scf = problem.scf
    hartree = problem.hartree
    eigensolver = problem.eigensolver
    mixing = problem.mixing
    boundary = _boundary(translation)
    lines: list[str] = [
        f"# {COMPLETE_INPUT_NAME}: the effective input of {translation.source}",
        "# (Generate_Complete_Parsec_In, as PARSEC 1.5): lengths in bohr, energies in Ry, temperatures in K,",
        "# the cell and the atoms after Super_Cell; the keywords below 'Other keywords of the input' are copied.",
        "",
        "# Real-space grid and domain",
        f"Boundary_Conditions  {boundary}",
    ]
    # paths that keep a comment character after _path_text (named in a warning of the header)
    unreadable: list[str] = []
    cell = problem.periodic_cell
    if cell is None:
        shape = getattr(grid, "domain_shape", "sphere")
        lines.append(f"Cluster_Domain_Shape  {shape}")
        if shape == "sphere":
            lines.append(f"Boundary_Sphere_Radius  {_number(grid.radius)} bohr")
        elif shape == "box":
            lines += ["begin Domain_Shape_Parameters", _row(grid.box_lengths), "end Domain_Shape_Parameters"]
        else:
            lines += [
                "begin Domain_Shape_Parameters",
                _row(grid.domain_parameters[:2]) + f"  {int(grid.domain_parameters[2])}"
                if shape == "cylinder"
                else _row(grid.domain_parameters),
                "end Domain_Shape_Parameters",
            ]
    lines += [
        f"Grid_Spacing  {_number(grid.spacing)} bohr",
        f"Expansion_Order  {int(grid.expansion_order)}",
        f"Ignore_Symmetry  {_logical(translation.ignore_symmetry)}",
    ]
    if boundary in ("cluster", "bulk"):
        lines.append(f"Double_Grid_Order  {int(getattr(grid, 'double_grid_order', 1))}")
    if cell is not None:
        vectors = np.asarray(cell.lattice_vectors, dtype=float)
        lines.append("Lattice_Vector_Scale  1.0 bohr")
        if boundary == "wire":
            rows = [vectors[0, :1]]
        elif boundary == "slab":
            rows = [vectors[0], vectors[1]]
        else:
            rows = list(vectors)
        lines += ["begin Cell_Shape", *(_row(row) for row in rows), "end Cell_Shape"]
        if boundary == "slab" and getattr(cell, "dimensions", 3) == 2:
            lines.append(f"Boundary_Sphere_Radius  {_number(0.5 * vectors[2, 2])} bohr")
        elif boundary in ("wire", "slab"):
            lines.append(f"Boundary_Sphere_Radius  {_number(grid.confinement_radius)} bohr")

    # Eigensolver (usrinputfile.F90 reads Chebdav_Degree for chebff/chebdav, FF_MaxIter for chebff, the
    # filter degrees with Chebyshev filtering, Chebyshev_Filtering for arpack/diagla).
    method = eigensolver.method
    filtering = method in ("chebff", "chebdav") or bool(eigensolver.chebyshev_filtering)
    lines += ["", "# Eigensolver", f"Eigensolver  {method}", f"Diag_Tolerance  {_number(eigensolver.tolerance)}"]
    if method in ("chebff", "chebdav"):
        lines.append(f"Chebdav_Degree  {int(eigensolver.first_filter_degree)}")
    if method == "chebff":
        lines.append(f"FF_MaxIter  {int(eigensolver.first_filter_cycles)}")
    if method in ("arpack", "diagla"):
        lines.append(f"Chebyshev_Filtering  {_logical(bool(eigensolver.chebyshev_filtering))}")
    # (stored whatever the solver, so written where the input gives them too)
    if filtering or "chebyshev_degree" in scalar or "chebyshev_degree_delta" in scalar:
        lines += [
            f"Chebyshev_Degree  {int(eigensolver.filter_degree)}",
            f"Chebyshev_Degree_Delta  {int(eigensolver.filter_degree_delta)}",
        ]
    lines.append(f"Matvec_Blocksize  {int(eigensolver.matvec_block_size)}")
    # Exact_State_Count (PARSEC.py; 1.5 has no buffer) in place of the buffer, which it does not use.
    if getattr(eigensolver, "exact_state_count", False):
        lines.append(f"Exact_State_Count  {_logical(True)}")
    else:
        lines.append(f"Subspace_Buffer_Size  {int(eigensolver.subspace_buffer)}")
    lines += [
        f"Dynamic_Diag_Tol  {_logical(bool(eigensolver.dynamic_tolerance))}",
    ]
    # Labels Andrew Grieder's routines do not read select the Fortran-derived ones (Input.routines.decide_routines):
    # they are written where the input gives them or their value is not the default.
    if "maximum_matvec" in scalar or int(eigensolver.maximum_matvec) != 30000:
        lines.append(f"Maximum_Matvec  {int(eigensolver.maximum_matvec)}")
    # The seed of the start vectors (a clock seed of Vector_Rng_Seed -1 as the
    # number drawn) and the residual test of the first filter, where set.
    if getattr(eigensolver, "vector_seed", None) is not None:
        lines.append(f"Vector_Rng_Seed  {int(eigensolver.vector_seed)}")
    if getattr(eigensolver, "first_filter_residual_test", "off") != "off":
        lines.append(f"FF_Residual_Test  {eigensolver.first_filter_residual_test}")

    # Self-consistency and mixing
    lines += [
        "",
        "# Self-consistency and mixing",
        f"Max_Iter  {int(scf.max_iterations)}",
        f"Convergence_Criterion  {_number(scf.convergence_criterion)} Ry",
    ]
    if scf.convergence_criterion_approach is not None:
        lines.append(f"Convergence_Criterion_Approach  {_number(scf.convergence_criterion_approach)} Ry")
    lines += [
        f"Use_Plain_SRE  {_logical(bool(scf.use_plain_residual))}",
        f"Mixing_Method  {mixing.method}",
        f"Mixing_Param  {_number(mixing.parameter)}",
        f"Memory_Param  {int(mixing.memory)}",
    ]
    # Each mixer reads its keywords; one the input gives for another mixer is kept (it is part of the settings).
    secant = mixing.method.startswith("msecant")
    for key, label, value, read in (
        ("restart_mixing", "Restart_Mixing", int(mixing.restart), mixing.method in ("anderson", "broyden")),
        ("mixing_safeguard", "Mixing_Safeguard", _logical(bool(mixing.safeguard)), mixing.method == "anderson"),
        ("mixing_step_limit", "Mixing_Step_Limit", _number(mixing.step_limit), bool(mixing.safeguard)),
        ("mixing_growth_trigger", "Mixing_Growth_Trigger", _number(mixing.growth_trigger), bool(mixing.safeguard)),
        ("mixing_backoff", "Mixing_Backoff", _number(mixing.backoff), bool(mixing.safeguard)),
        ("mixing_group_size", "Mixing_Group_Size", int(mixing.group_size), secant),
        ("mixing_en_like", "Mixing_EN_Like", int(mixing.en_like), secant),
        ("mixing_restart_factor", "Mixing_Restart_Factor", _number(mixing.restart_factor), secant),
        ("mixing_memory_expand_factor", "Mixing_Memory_Expand_Factor", _number(mixing.expand_factor), secant),
        ("mixing_preferred_type", "Mixing_Preferred_Type", int(mixing.preferred_type), mixing.method == "msecant3"),
    ):
        if read or key in scalar:
            lines.append(f"{label}  {value}")

    # Hartree potential
    lines += [
        "",
        "# Hartree potential",
        f"Solver_Lpole  {int(hartree.multipole_order)}",
        f"Full_Hartree  {_logical(hartree.boundary_method == 'direct')}",
    ]
    if cell is None:
        lines += [
            "Hartree_Boundary_Tolerance  "
            + ("off" if hartree.boundary_tolerance is None else f"{_number(hartree.boundary_tolerance)} Ry"),
            f"Hartree_Atomic_Tail  {hartree.atomic_tail}",
            f"Domain_Energy_Tolerance  {_number(translation.domain_energy_tolerance)} Ry",
        ]
    if "hartree_cg_tolerance" in scalar or "hartree_cg_criterion" in scalar:
        # written only where the input gives them: they select PARSEC 1.5's CG (Input.parsec_input)
        lines += [
            f"Hartree_CG_Tolerance  {_number(hartree.relative_tolerance)}",
            f"Hartree_CG_Criterion  {int(getattr(hartree, 'cg_criterion', 1))}",
        ]

    # Electrons and exchange-correlation
    lines += [
        "",
        "# Electrons and exchange-correlation",
        f"States_Num  {int(scf.number_of_states)}",
        f"Net_Charges  {_number(scf.net_charge)}",
        f"Fermi_Temp  {_number(scf.fermi_temperature_kelvin)}",
        f"Spin_Polarization  {_logical(bool(scf.spin_polarized))}",
        f"Correlation_Type  {FORTRAN_CORRELATION_CODES.get(scf.xc_functional, scf.xc_functional)}",
    ]
    if scf.spin_polarized and (scf.is_hybrid or getattr(scf, "spin_symmetry_sectors", None) is not None):
        # The spin channels of the accelerated driver (PARSEC.py): the value of
        # every spin-polarized hybrid, and of another run where the input sets
        # it away from the default (auto), so that the completed input of a
        # semilocal run at the default stays what it was.
        lines.append(f"Spin_Symmetry_Sectors  {scf.spin_symmetry_sector_mode}")
    if scf.xc_functional == "casida_salahub":
        lines.append(f"Ion_Energy_Diff  {_number(scf.ionization_energy_difference)} Ry")
    exact_exchange = getattr(scf, "exact_exchange", None)
    if exact_exchange is not None:
        # The Fock loop of a hybrid functional (PARSEC.py).
        if exact_exchange.fraction is not None:
            lines.append(f"Exact_Exchange_Fraction  {_number(exact_exchange.fraction)}")
        if exact_exchange.screening is not None:
            lines.append(f"Screening_Parameter  {_number(exact_exchange.screening)}")
        lines += [
            f"Exx_Max_Iter  {int(exact_exchange.max_iterations)}",
            f"Exx_Tolerance  {_number(exact_exchange.tolerance)} Ry",
            f"Exx_Batch_Size  {int(exact_exchange.batch_size)}",
            f"Exx_Pair_Solver  {exact_exchange.pair_solver}",
            f"Exx_Inner_Criterion  {exact_exchange.inner_criterion}",
            f"Exx_Mixer_History  {exact_exchange.mixer_history}",
            f"Exx_Outer_Mixing  {exact_exchange.outer_mixing}",
        ]
    if problem.point_charges:
        counts = tuple(getattr(translation, "point_charge_type_counts", ())) or (len(problem.point_charges),)
        lines += ["Add_Point_Charges  .true.", f"Point_Typ_Num  {len(counts)}"]
        start = 0
        for count in counts:
            group = problem.point_charges[start : start + count]
            start += count
            lines.append(f"Pt_Chg  {_number(group[0].charge)}")
            lines += ["begin Point_Coord", *(_row(item.position) for item in group), "end Point_Coord"]
    sheets = tuple(getattr(problem, "charged_sheets", ()))
    if sheets:
        lines += ["Add_Charged_Sheets  .true.", f"Num_Charged_Sheets  {len(sheets)}"]
        for sheet in sheets:
            lines += [
                f"Sheet_Charge  {_number(sheet.charge)}",
                "begin Sheet_Coord",
                _row((0.0, 0.0, sheet.height)),
                "end Sheet_Coord",
            ]

    # Structural relaxation (an AFM scan makes its own manual sequence: AFM_Scan below)
    relaxation = translation.relaxation
    scan = getattr(translation, "afm_scan", None)
    if relaxation is not None and scan is None:
        lines += [
            "",
            "# Structural relaxation",
            f"Minimization  {relaxation.method}",
            f"Movement_Num  {int(relaxation.max_movements)}",
            f"Force_Min  {_number(relaxation.minimum_force)} Ry/bohr",
            f"Max_Step  {_number(relaxation.maximum_step)} bohr",
            f"Min_Step  {_number(relaxation.minimum_step)} bohr",
            f"BFGS_Number_Corr  {int(relaxation.bfgs_corrections)}",
            f"Relax_Restart  {_logical(bool(relaxation.restart))}",
        ]
        if relaxation.scf_upscale != 1.0:
            lines.append(f"Relaxation_SCF_Upscale  {_number(relaxation.scf_upscale)}")

    lines += _fdet_lines(translation, unreadable)

    # The species and their atoms (Cartesian bohr; the movable flags where atoms move)
    moves = (
        translation.relaxation is not None and scan is None
    ) or translation.molecular_dynamics is not None
    movable = tuple(translation.movable)
    atoms = tuple(problem.atoms)
    if scan is not None:
        # the input tip, which the scan displaces (the problem holds its first point)
        from dataclasses import replace

        atoms = tuple(replace(atom, position=position) for atom, position in zip(atoms, scan.tip_positions))
    lines += [
        "",
        "# Species and atoms",
        "Coordinate_Unit  cartesian_bohr",
        f"Atom_Types_Num  {len(problem.pseudopotentials)}",
    ]
    ylm_cutoffs = list(scalar.get("ylm_dos_cutoff", ()))
    species_list = tuple(problem.pseudopotentials.values())
    hubbard = "ldaplusu_u" in scalar or "ldaplusu_j" in scalar or any(
        item.hubbard_u != 0.0 or item.hubbard_j != 0.0 for item in species_list
    )
    # A filter of one species makes the parser read all three of every species; Core_Filter defaults to Alpha_Filter.
    filters = any(
        item.fourier_filter_alpha != 0.0 or item.fourier_filter_beta1 != 0.0 or item.fourier_filter_core is not None
        for item in species_list
    )
    core_filters = any(item.fourier_filter_core is not None for item in species_list)
    for index, (symbol, species) in enumerate(problem.pseudopotentials.items()):
        pseudo_format = species.pseudopotential_format
        lines += [
            "",
            f"Atom_Type  {symbol}",
            f"Element_Symbol  {species.element_symbol or symbol}",
            f"Pseudopotential_Format  {pseudo_format}",
            "Pseudopot_File  "
            + _path_text(species.path, translation.source.parent, f"Pseudopot_File of {symbol}", unreadable),
        ]
        if pseudo_format != "martins_wang":
            lines.append(f"Local_Component  {_LOCAL_LABELS[int(species.local_angular_momentum)]}")
        if pseudo_format in ("martins_old", "fhipp"):
            lines.append(f"Potential_Num  {int(species.number_of_channels)}")
        if pseudo_format in ("martins_old", "martins_wang", "fhipp"):
            lines.append(f"Core_Cutoff_Radius  {_number(species.core_cutoff_radius)} bohr")
        if pseudo_format in ("martins_old", "fhipp"):
            lines += [
                "begin Electron_Per_Orbital",
                _row(species.electrons_per_orbital),
                "end Electron_Per_Orbital",
            ]
        if pseudo_format == "fhipp":
            lines.append(f"Nonlinear_Core_Correction  {_logical(bool(species.nonlinear_core_correction))}")
        lines += [
            f"SO_PSP  {_logical(bool(species.spin_orbit))}",
            f"Read_VCD  {_logical(bool(species.read_valence_density))}",
            f"Cubic_Spline  {_logical(bool(species.use_spline))}",
            f"Initial_Spin_Polarization  {_number(species.initial_spin_polarization)}",
            f"Atomic_Energy_Correction  {_number(species.atomic_energy_correction)} Ry",
        ]
        # (labels Andrew Grieder's routines do not read: where the input gives them, see Maximum_Matvec)
        if hubbard:
            lines += [f"ldaplusu_u  {_number(species.hubbard_u)}", f"ldaplusu_j  {_number(species.hubbard_j)}"]
        if problem.scf.noncollinear:
            lines.append(f"Atomic_Radius  {_number(species.atomic_radius)}")
        if cell is not None and filters:
            lines += [
                f"Alpha_Filter  {_number(species.fourier_filter_alpha)}",
                f"Beta1_Filter  {_number(species.fourier_filter_beta1)}",
            ]
            if core_filters:
                # the n-th Core_Filter goes to the n-th species: every species takes one (its default is Alpha_Filter)
                core = species.fourier_filter_alpha if species.fourier_filter_core is None else species.fourier_filter_core
                lines.append(f"Core_Filter  {_number(core)}")
        if index < len(ylm_cutoffs):
            lines.append(f"Ylm_DOS_Cutoff  {ylm_cutoffs[index].value}")
        own = [position for position, atom in enumerate(atoms) if atom.symbol == symbol]
        if moves:
            lines.append("Move_Flag  -1")
        lines.append("begin Atom_Coord")
        for position in own:
            row = _row(atoms[position].position)
            if moves:
                row += f"  {1 if movable[position] else 0}"
            lines.append(row)
        lines.append("end Atom_Coord")
        if problem.scf.noncollinear:
            lines.append("begin Initial_NCL_Moment")
            for position in own:
                moment = atoms[position].initial_moment
                lines.append(_row((0.0, 0.0, 0.0) if moment is None else moment))
            lines.append("end Initial_NCL_Moment")

    lines += [
        "",
        "# Output",
        f"Output_Level  {int(translation.output_level)}",
        f"Output_All_States  {_logical(bool(translation.output_all_states))}",
    ]
    # The routines of the run (Input.routines): the expanded Super_Cell or a keyword written here could choose
    # others, so the choice is written where it is not the default.
    routines = getattr(translation, "routines", None)
    if routines is not None and routines.requested is not None:
        lines += ["", "# Routines", f"Use_Fortran_Routines  {_logical(routines.requested)}"]
    elif routines is not None and routines.fortran:
        chosen = ", ".join((*routines.defaults, *routines.reasons)).replace(";", ",")
        lines += ["", "# Routines (the run chose them: " + chosen.replace("#", "") + ")", "Use_Fortran_Routines  .true."]

    # The k-point lists of an input whose Super_Cell is expanded here (the reader would convert them with the
    # expanded cell): their Cartesian k-points.
    kpoint_lines, converted = _kpoint_list_lines(translation, scalar, blocks, kpoint_cell)
    lines += kpoint_lines

    # Everything else, as the input gives it (in its order)
    skipped = _RECONSTRUCTED | converted
    others = sorted(
        [item for key, items in scalar.items() if key not in skipped for item in items]
        + [item for key, items in blocks.items() if key not in skipped for item in items],
        key=lambda item: item.line,
    )
    if others:
        lines += ["", "# Other keywords of the input"]
        for item in others:
            if item.is_block:
                lines += [f"begin {item.key}", *(f"  {line}" for line in item.value), f"end {item.key}"]
            else:
                lines.append(f"{item.key}  {item.value}")
    if unreadable:
        # ESDF (PARSEC 1.5 and this reader) ends a line at '#', ';' or '!' and has no quoting
        named = ", ".join(unreadable)
        lines[3:3] = [
            f"# WARNING: this file does not read back as it is: {named} hold a comment character",
            "# WARNING: (#, ; or !), which ends an input line for PARSEC's ESDF reader and this one (no quoting);",
            "# WARNING: give the file a path without one before reading this file.",
        ]
    return "\n".join(lines) + "\n"


def _kpoint_list_lines(translation, scalar, blocks, kpoint_cell) -> tuple[list[str], frozenset[str]]:
    """``Kpoint_List`` and ``nscf_kpoints`` of an input whose ``Super_Cell`` the bak file expands, in 1/bohr.

    The reader converts ``reciprocal_lattice_vectors`` with the input cell where a ``Super_Cell`` or
    ``Super_Cell_Vac`` is given (``Input.parsec_input``, as usrinputfile.F90 does with ``lattvec``); the bak file has
    the expanded cell and no ``Super_Cell``, so its lists are written Cartesian: the k-points the reader made of the
    input's rows (with ``kpoint_cell``), bit for bit, beside the weights of the input (normalised again by the
    reader, to the same numbers).  A ``Kpoint_File`` becomes a ``Kpoint_List``.  Monkhorst-Pack grids and their
    shifts, the bandstruc path and the DOS grid are fractions of the cell the run uses, the expanded one, and are
    copied.  Returns the lines and the labels they replace (none without an expanded ``Super_Cell``).
    """

    from pathlib import Path

    from ..Grid.kpoints import explicit_kpoints, read_kpoint_file
    from .parsec_input import _block_numbers, _normalize_label, _unquoted

    problem = translation.problem
    expanded = problem.periodic_cell is not None and (
        getattr(translation, "super_cell", None) is not None
        or getattr(translation, "super_cell_vacuum", None) is not None
    )
    if not expanded or kpoint_cell is None:
        return [], frozenset()

    def first(key: str, default: str) -> str:
        items = scalar.get(key, ())
        return str(items[0].value) if items else default

    def rows_of(block) -> np.ndarray:
        rows = []
        for line in block.value:
            values, _unit = _block_numbers((line,), label=block.key)
            rows.append(list(values))
        return np.asarray(rows, dtype=float)

    def written(raw: np.ndarray, unit: str) -> list[str]:
        points, _weights = explicit_kpoints(raw, unit=unit, cell=kpoint_cell)
        if raw.shape[1] == 3:
            return [_row(point) for point in points]
        return [_row(point) + f"  {_number(weight)}" for point, weight in zip(points, raw[:, 3])]

    lines: list[str] = []
    converted: set[str] = set()
    if _normalize_label(first("kpoint_method", "none")) == "manual":
        if blocks.get("kpoint_list"):
            raw = rows_of(blocks["kpoint_list"][0])
        else:
            candidate = Path(_unquoted(first("kpoint_file", ""))).expanduser()
            if not candidate.is_absolute():
                candidate = Path(translation.source).parent / candidate
            raw = read_kpoint_file(candidate.resolve())
        unit = _normalize_label(first("kpoint_unit", "cartesian_inverse_bohr"))
        lines += [
            "",
            "# k-points of the input in 1/bohr (its Super_Cell is expanded above)",
            "Kpoint_Method  manual",
            "Kpoint_Unit  cartesian_inverse_bohr",
            "begin Kpoint_List",
            *written(raw, unit),
            "end Kpoint_List",
        ]
        converted |= {"kpoint_method", "kpoint_unit", "kpoint_list", "kpoint_file"}
    if blocks.get("nscf_kpoints"):
        raw = rows_of(blocks["nscf_kpoints"][0])
        unit = _normalize_label(first("nscf_kpoint_unit", "cartesian_inverse_bohr"))
        lines += [
            "",
            "# non-self-consistent k-points of the input in 1/bohr (its Super_Cell is expanded above)",
            "nscf_kpoint_unit  cartesian_inverse_bohr",
            "begin nscf_kpoints",
            *written(raw, unit),
            "end nscf_kpoints",
        ]
        converted |= {"nscf_kpoint_unit", "nscf_kpoints"}
    return lines, frozenset(converted)


__all__ = ["COMPLETE_INPUT_NAME", "complete_input_text"]
