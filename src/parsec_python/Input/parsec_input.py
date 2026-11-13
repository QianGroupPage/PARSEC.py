"""Focused ESDF/PARSEC input reader for isolated CA-LDA single points."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
from time import perf_counter
from typing import TYPE_CHECKING, Iterable

import numpy as np

from ..models import (
    DEFAULT_INITIAL_SPIN_POLARIZATION,
    DEFAULT_RELAXATION_MAXIMUM_STEP,
    Atom,
    BandSegment,
    BandStructureSettings,
    ChargedSheet,
    DOSSettings,
    GRID_DATA_FIELDS,
    EigensolverSettings,
    GWSettings,
    GridSettings,
    HartreeSettings,
    InitialDensitySettings,
    MIXING_METHODS,
    MIXING_PRECONDITIONERS,
    MixingSettings,
    NSCFSettings,
    OutputFileSettings,
    RelaxationSettings,
    WavefunctionSelection,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
    PeriodicCell,
    PeriodicGridSettings,
    PointCharge,
    UNPOLARIZED_ONLY_XC_FUNCTIONALS,
    ExactExchangeSettings,
    HYBRID_XC_FUNCTIONALS,
)
from ..models import EmbeddingSettings  # FDET of PARSEC 1.5
from ..Grid.cell import primitive_lattice, replicate_positions, supercell_lattice
from ..Grid.pbc import fortran_grid_shift
from ..Grid.kpoints import explicit_kpoints, read_kpoint_file
from ..Grid.pbc import periodic_grid_shift, uses_general_periodic_grid
from . import routines as _routines_module
from .routines import RoutineChoice, UseFortranRoutines, decide_routines
from .solver_options import SOLVER_KEYS as _SOLVER_1P5_KEYS, read_solver_options
from ..MLDensity.field import normalize_density_units
from ..Pseudopotential.potrw import read_martins_wang_header
from ..MolecularDynamics.settings import MolecularDynamicsSettings

if TYPE_CHECKING:
    from ..Hartree.domain import DomainChoice
    from ..MLFF.settings import MLFFSettings


# PARSEC's ESDF table uses 1 bohr = 0.529177 angstrom exactly.
ANGSTROM_TO_BOHR = 1.0 / 0.529177
EV_TO_RYDBERG = 1.0 / 13.605693122994
# Energies of ESDF's table in joule (esdf_mod.f90, phy_u), for the keywords
# whose conversion follows the Fortran exactly (ldaplusu_u, ldaplusu_j).
_ESDF_ENERGY_JOULE = {
    "ry": 2.17991e-18,
    "mry": 2.17991e-21,
    "ev": 1.60219e-19,
    "mev": 1.60219e-22,
    "hartree": 4.35982e-18,
    "ha": 4.35982e-18,
    "mhartree": 4.35982e-21,
}
# ESDF's ``k`` (esdf_mod.f90 phy_n(54)): an energy unit, k_B * 1 K in joule.
_ESDF_BOLTZMANN_JOULE = 1.38066e-23

_FLOAT_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
)
_COMMENT_MARKERS = ("#", ";", "!")
# Correlation_Type labels of usrinputfile.F90:2328-2361 (after esdf_reduce)
# and the functional of SCFSettings.xc_functional each one selects.
_CORRELATION_LABELS = {
    "xa": "xalpha",
    "wi": "wigner",
    "hl": "hedin_lundqvist",
    "ca": "ca",
    "pz": "ca",
    "lda": "ca",
    "pl": "pw92",
    "pw92": "pw92",
    "pwlda": "pw92",
    "pw": "pw91",
    "pw91": "pw91",
    "pwgga": "pw91",
    "pb": "pbe",
    "pbe": "pbe",
    # PARSEC 1.5 (usrinputfile.F90:4049-4054): PBE with the variational
    # White-Bird potential, which is this port's PBE (see _parse_parsec_input).
    "wb": "pbe",
    "lb": "lb94",
    "cs": "casida_salahub",
}
# Meta-GGAs of this package (PARSEC.py; PARSEC 1.5 reaches meta-GGAs through
# libxc only): Correlation_Type label -> SCFSettings.xc_functional
# (parsec_python.V_xc.meta_gga).
_META_GGA_LABELS = {
    "scan": "scan",
}
# Self-consistent hybrid functionals (PARSEC.py; no Fortran PARSEC label), the
# Fock loop of parsec_python.SCF.hybrid: the labels and aliases of
# parsec_python.V_xc.hybrid (canonical_hybrid_label), read before the table above.
# The Fock-loop keywords of a hybrid Correlation_Type (PARSEC.py).
_HYBRID_KEYS = (
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
)


class ParsecInputError(ValueError):
    """Invalid or unsupported input in the scoped PARSEC reader."""


class UnsupportedParsecOptionError(ParsecInputError):
    """A valid PARSEC feature lies outside this single-point implementation."""


@dataclass(frozen=True)
class ParsecInputTranslation:
    """Translated Python problem plus nonfatal compatibility notes."""

    source: Path
    problem: SinglePointInput
    warnings: tuple[str, ...]
    output_all_states: bool
    output_level: int
    ignore_symmetry: bool = False
    # PARSEC 1.5's ``Grid_Shift`` block (its absolute values, before the zero
    # along oblique periodic directions), ``None`` without it; the grid
    # settings of the problem hold the shift the run uses.
    grid_shift: tuple[float, float, float] | None = None
    # Whether an atom of the input lies outside the periodic cell [0, 1) of
    # the preparation; ``None`` where the parser did not translate the atoms
    # at start (PARSEC 1.5's back_to_cell, the Fortran-derived routines).
    # The report notes translated coordinates only for those.
    atoms_outside_cell: bool | None = None
    # Domain_Energy_Tolerance in Ry.  ``domain`` is what the default rule
    # chose where the input leaves the sphere radius to it, and ``None`` for
    # an input with a radius or a box.  ``boundary_tolerance_from`` says
    # where the Hartree boundary tolerance of the problem comes from: its
    # own line ("input"), the fixed default beside a radius ("default"), or
    # the rule ("rule").  ``domain_rule_seconds`` is what the rule took.
    domain_energy_tolerance: float = 1.0e-3
    domain: "DomainChoice | None" = None
    boundary_tolerance_from: str = "default"
    domain_rule_seconds: float = 0.0
    # Structural relaxation (``Minimization``), ``None`` for a single point.
    # ``movable`` holds one flag per atom of the problem (``Move_Flag`` and
    # the optional fourth column of ``Atom_Coord``).  ``compute_forces`` says
    # that the forces of the structure are wanted: always with a relaxation
    # or dynamics, and for a single point when the input says
    # ``Skip_Force: false``.  ``use_symmetry`` is false where PARSEC switches
    # the point group off although ``Ignore_Symmetry`` is not set: manual
    # movements and molecular dynamics.
    relaxation: "RelaxationSettings | None" = None
    movable: tuple[bool, ...] = ()
    compute_forces: bool = False
    use_symmetry: bool = True
    # Molecular dynamics (``Molecular_Dynamics``), ``None`` when it is off.
    molecular_dynamics: "MolecularDynamicsSettings | None" = None
    # ``Potential_Extrapolation``: how the SCF loop of a moved structure
    # starts.  ``none`` as PARSEC (the last potential with the ionic
    # potential exchanged), ``atomic`` with the superposed atomic densities
    # moved along with the atoms, ``second_order`` with the change of the
    # density extrapolated along the trajectory
    # (:mod:`parsec_python.SCF.single_point`).
    potential_extrapolation: str = "none"
    # ``MD_Output_Interval`` (PARSEC.py extension): md_mech.dat, atom.cor, movie.xyz, md_frames.extxyz and the force
    # table of parsec.out are written every this many MD steps and at the last one; md_nrg.dat every step.
    md_output_interval: int = 1
    # ``Initial_Structure: mlff``: the input structure is relaxed with a
    # machine-learned force field before the run (:mod:`parsec_python.MLFF`),
    # ``None`` when the run starts from the coordinates of the input.
    mlff: "MLFFSettings | None" = None
    # ``Energy_Refinement``: the hybrid functional whose exchange-correlation
    # energy is evaluated once on the converged orbitals after the SCF
    # (:mod:`parsec_python.Energy.refinement`), ``None`` when it is off.  It
    # is a reporting step and is not part of the problem the SCF solves.
    energy_refinement: str | None = None
    # ``Polarizability``: the field F of the seven finite-field SCF loops of
    # polar.f90 (``Electric_Field`` in ry/bohr/e, used as PARSEC uses it:
    # V = 2 F d.r, see parsec_python.SCF.polarizability), ``None`` when off.
    polarizability_field: float | None = None
    # ``Point_Typ_Num`` groups of the point charges of the problem: the
    # number of charges of each ``Pt_Chg``/``Point_Coord`` type, in order.
    point_charge_type_counts: tuple[int, ...] = ()
    # Inert keywords of PARSEC, kept for the lines it prints: the value of
    # ``External_Mag_Field`` divided by 2.5e9 (what PARSEC prints in Gauss;
    # it applies no field) and whether the wavefunctions are declared
    # complex (``Complex_Wfn``, or forced by a magnetic field).
    magnetic_field_gauss: float = 0.0
    complex_wavefunctions: bool = False
    # ``parsec.dat``, ``eigen.dat``, cube files and ``Output_File_Name``
    # (parsec_python.Output.output_files); they change no number of the run.
    output_files: OutputFileSettings = OutputFileSettings()
    # ``Restart_Run``: the SCF loop starts from the potential of parsec.dat
    # beside the input (parsec_python.SCF.restart).
    restart_run: bool = False
    # ``Restart_From_Wfndat`` with ``Restart_Run``: the restart reads the
    # legacy wfn.dat (potential, density and wavefunctions) instead of
    # parsec.dat (restart.F90; parsec_python.Output.wfn_dat).
    restart_from_wfndat: bool = False
    # ``Fix_Number_Eigenvalues`` of a restart (usrinputfile.F90:1444-1449):
    # the eigensolver keeps the number of eigenpairs of the restart file
    # (EigensolverSettings.fixed_state_count); True also for an NSCF run,
    # where calc_nscf.F90 sets the number itself (the lines are printed).
    fix_number_eigenvalues: bool = False
    # ``Dynamic_Diag_Tol`` given together with ``Restart_Run``:
    # usrinputfile.F90:1492-1495 switches it off and says so.
    dynamic_tolerance_off_for_restart: bool = False
    # ``Super_Cell`` and ``Super_Cell_Vac`` of a periodic input (``None``
    # without the block): the replication and the vacuum factors applied to
    # the lattice vectors, for the report.
    super_cell: tuple[int, int, int] | None = None
    super_cell_vacuum: tuple[int, int, int] | None = None
    # ``nscf_kpoints``: a non-self-consistent run from the potential of
    # parsec.dat instead of the SCF (parsec_python.SCF.nscf), ``None`` without.
    nscf: NSCFSettings | None = None
    # ``Create_DOS``: the DOS file after the SCF (parsec_python.Output.dos).
    dos: DOSSettings | None = None
    # ``Output_GW``: the BerkeleyGW files WFN, RHO and VXC after the SCF or
    # the NSCF run (parsec_python.Output.berkeleygw), ``None`` without.
    gw: GWSettings | None = None
    # The z shift (bohr) PARSEC's initial.f90:201-228 subtracted from every
    # atom of a slab to centre it in its box (0.0: not shifted); the problem
    # holds the shifted atoms.
    slab_shift: float = 0.0
    # The routines of the run (parsec_python.Input.routines): Andrew
    # Grieder's (the default) or the Fortran-derived ones
    # (``Use_Fortran_Routines``, or an input the default does not compute).
    routines: RoutineChoice = RoutineChoice()
    # ``Write_Potential_Field`` (PARSEC.py): pot.dat of PARSEC 1.5 after the
    # run (parsec_python.Output.pot_dat); 1.5 writes it after every run.
    write_potential_field: bool = False
    # ``AFM_Scan`` (PARSEC.py): the scan of an FDET tip (parsec_python.Embedding.afm),
    # run as Minimization: manual over its structures; ``None`` without it.
    afm_scan: object | None = None
    # ``Generate_Complete_Parsec_In`` (PARSEC 1.5): the text of
    # bak_complete_parsec.in, the effective input of the run
    # (parsec_python.Input.complete_input), which the CLIs write beside the
    # input; ``None`` without the keyword.
    complete_input: str | None = None
    # PARSEC 1.5 ``Recenter_Atoms_In_Aperiodic_Directions``: False keeps the
    # coordinates of the input also where PARSEC re-centres them (a cluster,
    # a slab along z, the first structure of a molecular dynamics restart).
    recenter_aperiodic: bool = True


@dataclass(frozen=True)
class _InputItem:
    key: str
    value: str | tuple[str, ...]
    line: int
    is_block: bool = False


#: Keywords of ``usrinputfile.F90`` for molecular dynamics.
_MOLECULAR_DYNAMICS_KEYS = (
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
)

# ---------------------------------------------------------------------------
# PARSEC 1.5 keywords of the forces, ionic loops and band structure (parity
# round of 2026-11-04; src/io/usrinputfile.F90 of 1.5).  Read in
# :func:`_small15_forces_requested`, :func:`_small15_ionic_keywords` and, in
# :func:`_parse_parsec_input`, the bandstruc block, the relaxation settings and
# the re-centring of clusters and slabs.
# ---------------------------------------------------------------------------
_SMALL15_ACCEPTED_GLOBAL = frozenset(
    {
        # usrinputfile.F90:4151, parsec.F90:2422-2505: forces of a single point
        "calculate_force",
        # parsec.F90:2436,2477: the serial force routines (the same forces)
        "use_old_forces",
        # domove.f90:390: relax_restart.dat
        "write_relax_restart",
        # usrinputfile.F90:3653-3657, initial.f90:202-327
        "recenter_atoms_in_aperiodic_directions",
        # usrinputfile.F90:2243, calc_bands.f90:336-338
        "bands_ff_maxiter",
    }
)

#: ``Boundary_Conditions`` labels and their number of periodic directions
#: (``pbc%per``, ``usrinputfile.F90:392-411``).
_BOUNDARY_PERIODICITY = {
    "cluster": 0,
    "0d": 0,
    "wire": 1,
    "tube": 1,
    "1d": 1,
    "slab": 2,
    "2d": 2,
    "bulk": 3,
    "3d": 3,
    "pbc": 3,
}


# ---------------------------------------------------------------------------
# PARSEC 1.5 keywords of the input, I/O and physics layer (parity round of
# 2026-11-04; src/io/usrinputfile.F90 of 1.5).  They are kept in this block
# and in :func:`_input15_keywords`, beside the keywords of the forces, the
# grid, the eigensolvers and FDET.
# ---------------------------------------------------------------------------

#: Keywords PARSEC 1.5 renamed: its name -> the name this reader keeps
#: (``usrinputfile.F90:4741`` ``mol_rng_seed``, formerly ``rng_seed``).  Both
#: names are read; an input that gives both is refused.
_INPUT15_ALIASES = {"mol_rng_seed": "rng_seed"}

#: Keywords that only tune PARSEC 1.5's MPI layout, ScaLAPACK, ELPA, HDF5,
#: space-filling curves, spectrum slicing, threads or its debug tests, with
#: the name printed in the warning.  They change no number of this port.
_INPUT15_INERT_KEYS = {
    # usrinputfile.F90:2452-2492
    "use_scalapack": "Use_ScaLAPACK",
    "scalapack_block_size": "ScaLAPACK_Block_Size",
    "use_elpa": "Use_ELPA",
    "elpa_block_size": "ELPA_Block_Size",
    # usrinputfile.F90:4843-4897, the MPI column groups (MPI_Groups before 1.5)
    "npcol": "Npcol",
    # usrinputfile.F90:2366-2440, spectrum slicing
    "spectrum_slicing": "Spectrum_Slicing",
    "use_krylov_subspace_method": "Use_Krylov_Subspace_Method",
    "use_lanczos_dos_estimation": "Use_Lanczos_DOS_Estimation",
    "spectrum_slicing_cycles_early_quit": "Spectrum_Slicing_Cycles_Early_Quit",
    "spectrum_slicing_max_cycles": "Spectrum_Slicing_Max_Cycles",
    "spectrum_slicing_max_cycles_first": "Spectrum_Slicing_Max_Cycles_First",
    "num_of_iter_still_using_chebyshev": "Num_Of_Iter_Still_Using_Chebyshev",
    "polym_first_ss_iter": "Polym_First_SS_Iter",
    "polym_ss": "Polym_SS",
    "num_of_copied_groups": "Num_Of_Copied_Groups",
    "num_of_slice1_filtering": "Num_Of_Slice1_Filtering",
    "nslices": "Nslices",
    "spectrum_slicing_ratio": "Spectrum_Slicing_Ratio",
    # usrinputfile.F90:2510, the order of the grid points
    "use_sfc": "Use_Sfc",
    # usrinputfile.F90:2235, threaded random start vectors
    "thread_random_init": "Thread_Random_Init",
    # usrinputfile.F90:2667-2735: stored in rritz%forced_logic, read nowhere
    "rritz_logic": "Rritz_Logic",
    # usrinputfile.F90:2768-2784 (only 'none' is accepted here)
    "test_logic": "Test_Logic",
    # esdf_key_mod.f90: registered, never read by usrinputfile.F90
    "max_taskgroups": "Max_Taskgroups",
    "vxc_matrix_elements": "Vxc_Matrix_Elements",
    # usrinputfile.F90:4178-4200: HDF5 restart files instead of parsec.dat
    "use_hdf5": "Use_HDF5",
}

#: The ESDF type of the inert keywords: what usrinputfile.F90 reads them as.
_INPUT15_INERT_TYPES = {
    "use_scalapack": "logical",
    "scalapack_block_size": "integer",
    "use_elpa": "logical",
    "elpa_block_size": "integer",
    "npcol": "integer",
    "spectrum_slicing": "logical",
    "use_krylov_subspace_method": "logical",
    "use_lanczos_dos_estimation": "logical",
    "spectrum_slicing_cycles_early_quit": "logical",
    "spectrum_slicing_max_cycles": "integer",
    "spectrum_slicing_max_cycles_first": "integer",
    "num_of_iter_still_using_chebyshev": "integer",
    "polym_first_ss_iter": "integer",
    "polym_ss": "integer",
    "num_of_copied_groups": "integer",
    "num_of_slice1_filtering": "integer",
    "nslices": "integer",
    "spectrum_slicing_ratio": "double",
    "use_sfc": "logical",
    "thread_random_init": "logical",
    "rritz_logic": "text",
    "test_logic": "text",
    "max_taskgroups": "integer",
    "vxc_matrix_elements": "text",  # 'T:E' in esdf_key_mod.f90
    "use_hdf5": "logical",
}

#: Rritz_Logic values of usrinputfile.F90:2667-2735 (case-sensitive there).
_INPUT15_RRITZ_LOGIC = ("classic", "tiled", "dist1", "dist2", "distmixed", "custom", "none")

#: The other PARSEC 1.5 keywords of this layer that the reader accepts.
_INPUT15_ACCEPTED_GLOBAL = frozenset(
    {
        *_INPUT15_INERT_KEYS,
        *_INPUT15_ALIASES,
        # usrinputfile.F90:3895-3911: libxc (see _input15_keywords), and this port's names of it
        "use_libxc",
        "func_id_x",
        "func_id_c",
        "libxc_functional",
        "libxc_enforce_fhc",
        "libxc_density_threshold",
        "libxc_tau_threshold",
        # usrinputfile.F90:2836-2842: the Poisson CG of potential/hartree.F90
        "hartree_cg_tolerance",
        "hartree_cg_criterion",
        # usrinputfile.F90:4968-4969 and 5477-6040: bak_complete_parsec.in
        "generate_complete_parsec_in",
        # usrinputfile.F90:2183-2188: the ASCII dumps of io/plaintext_io.F90
        "output_ascii_data",
        "print_unconverged_density",
    }
)

#: Per-species keywords of PARSEC 1.5 (read once per Atom_Type, ESDF cursor).
_INPUT15_SPECIES_KEYS = frozenset({"pseudopot_file"})


def _hybrid_label(text: str) -> str | None:
    """The registry label of a hybrid ``Correlation_Type``, ``None`` for another one."""

    from ..V_xc.hybrid import canonical_hybrid_label

    raw = str(text).strip().lower()
    for candidate in (raw, raw.replace("-", "_"), raw.replace("_", "-")):
        try:
            label = canonical_hybrid_label(candidate)
        except ValueError:
            continue
        return label if label in HYBRID_XC_FUNCTIONALS else None
    return None


def _exact_exchange_settings(scalar, one, xc_functional: str) -> ExactExchangeSettings | None:
    """``Exact_Exchange_Fraction``, ``Screening_Parameter`` and the ``Exx_*`` keys.

    ``None`` for a functional without exact exchange, where the keys are
    refused; for a hybrid every key that is not given keeps its default
    (:class:`parsec_python.models.ExactExchangeSettings`).
    """

    given = [key for key in _HYBRID_KEYS if key in scalar]
    if not SCFSettings(xc_functional=xc_functional).is_hybrid:
        if given:
            raise ParsecInputError(
                ", ".join(given)
                + " belong to a hybrid Correlation_Type ("
                + ", ".join(sorted(HYBRID_XC_FUNCTIONALS))
                + f"), not to {xc_functional!r}"
            )
        return None
    keywords: dict = {}
    try:
        if "exact_exchange_fraction" in scalar:
            keywords["fraction"] = _plain_float(one("exact_exchange_fraction"), "Exact_Exchange_Fraction")
        if "screening_parameter" in scalar:
            keywords["screening"] = _inverse_length_bohr(one("screening_parameter"))
        if "exx_max_iter" in scalar:
            value = _plain_float(one("exx_max_iter"), "Exx_Max_Iter")
            if value != int(value):
                raise ValueError("Exx_Max_Iter must be an integer")
            keywords["max_iterations"] = int(value)
        if "exx_tolerance" in scalar:
            keywords["tolerance"] = _energy_rydberg(one("exx_tolerance"), label="Exx_Tolerance")
        if "exx_batch_size" in scalar:
            value = _plain_float(one("exx_batch_size"), "Exx_Batch_Size")
            if value != int(value):
                raise ValueError("Exx_Batch_Size must be an integer")
            keywords["batch_size"] = int(value)
        if "exx_pair_solver" in scalar:
            keywords["pair_solver"] = _normalize_label(one("exx_pair_solver"))
        for key, field in (
            ("exx_inner_criterion", "inner_criterion"),
            ("exx_mixer_history", "mixer_history"),
            ("exx_outer_mixing", "outer_mixing"),
        ):
            if key in scalar:
                keywords[field] = _normalize_label(one(key))
        for key, field in (("exx_divergence", "divergence"), ("exx_slab_coulomb", "slab_coulomb")):
            # periodic cells (parsec_python.SCF.hybrid_periodic): labels with dashes; auto is the default
            if key in scalar:
                value = str(one(key)).strip().lower().replace("_", "-")
                if not (key == "exx_divergence" and value == "auto"):
                    keywords[field] = value
        if "exx_auxiliary_exponent" in scalar:
            keywords["auxiliary_exponent"] = _plain_float(one("exx_auxiliary_exponent"), "Exx_Auxiliary_Exponent")
        settings = ExactExchangeSettings(**keywords)
    except ValueError as error:
        raise ParsecInputError(str(error)) from error
    from ..SCF.hybrid import hybrid_definition, require_pair_solver

    probe = SCFSettings(xc_functional=xc_functional, exact_exchange=settings)
    try:
        require_pair_solver(hybrid_definition(probe))
    except NotImplementedError as error:
        raise UnsupportedParsecOptionError(str(error)) from error
    except ValueError as error:
        raise ParsecInputError(str(error)) from error
    return settings


def _spin_symmetry_sectors(scalar, one, *, spin_polarized: bool, warnings: list) -> str | None:
    """``Spin_Symmetry_Sectors``: ``auto`` or ``off``, ``None`` where the input gives the run's default.

    The default is ``auto`` for every ``Correlation_Type``
    (:attr:`parsec_python.models.SCFSettings.spin_symmetry_sector_mode`); a
    value equal to it is stored as ``None``, so that the settings of a run at
    the default keep their text (cache keys) and the completed input reads
    back as the same problem.  Without ``Spin_Polarization`` the keyword has
    nothing to decide and is ignored with a warning.
    """

    if "spin_symmetry_sectors" not in scalar:
        return None
    value = _normalize_label(one("spin_symmetry_sectors"))
    if value not in ("auto", "off"):
        raise ParsecInputError(f"Spin_Symmetry_Sectors must be auto or off, got {one('spin_symmetry_sectors')!r}")
    if not spin_polarized:
        warnings.append(
            "Spin_Symmetry_Sectors is set, but the run is not spin-polarized (Spin_Polarization); it is ignored."
        )
        return None
    return None if value == "auto" else value


def _plain_float(text: str, label: str) -> float:
    match = _FLOAT_PATTERN.match(str(text).strip())
    if match is None or str(text).strip()[match.end():].strip():
        raise ValueError(f"{label} must be a number, got {text!r}")
    return float(match.group(0).replace("d", "e").replace("D", "e"))


def _inverse_length_bohr(text: str, label: str = "Screening_Parameter") -> float:
    """``label`` (``Screening_Parameter``) in bohr^-1: a number, optionally ``bohr^-1`` or ``ang^-1``."""

    value = str(text).strip()
    match = _FLOAT_PATTERN.match(value)
    if match is None:
        raise ValueError(f"{label} must be a number, got {text!r}")
    number = float(match.group(0).replace("d", "e").replace("D", "e"))
    unit = re.sub(r"\s+", "", value[match.end():].lower())
    if unit in ("", "bohr^-1", "1/bohr", "bohr-1"):
        return number
    if unit in ("ang^-1", "1/ang", "ang-1", "angstrom^-1", "1/angstrom"):
        return number / ANGSTROM_TO_BOHR
    raise ValueError(f"{label}: unknown unit {unit!r} (bohr^-1 or ang^-1)")


def _check_meta_gga_request(
    scf,
    *,
    is_periodic: bool,
    specifications,
    energy_refinement,
    embedding: bool,
    polarizability: bool,
    nscf: bool,
    output_gw: bool,
    bands: bool,
    restart: bool = False,
) -> None:
    """Refuse by name what the meta-GGAs do not cover yet (:mod:`parsec_python.V_xc.meta_gga`)."""

    hubbard = any(
        specification.hubbard_u != 0.0 or specification.hubbard_j != 0.0
        for specification in specifications.values()
    )
    refused = [
        label
        for condition, label in (
            # A meta-GGA hybrid (libxc's hyb_mgga_): the Fock loop with the meta-GGA loop as its
            # inner loops; the inner loops restart their mixer (the joined (V, vtau) history).
            (
                scf.is_hybrid and getattr(scf.exact_exchange, "mixer_history", "restart") == "carry",
                "Exx_Mixer_History: carry (the inner loops of a meta-GGA hybrid restart their mixer)",
            ),
            # Periodic meta-GGAs run the k-point and Gamma loops of SCF/meta_gga_periodic.py; the
            # exact exchange of periodic cells is not joined with them.
            (scf.is_hybrid and is_periodic, "a periodic Boundary_Conditions with a meta-GGA hybrid"),
            (
                scf.self_consistent_spin_orbit
                or any(specification.spin_orbit for specification in specifications.values()),
                "spin-orbit coupling (SO_PSP, SO_from_scratch)",
            ),
            (scf.noncollinear, "non-collinear magnetism"),
            (embedding, "FDET (Potential_Field)"),
            (hubbard, "LDA+U (ldaplusu_u, ldaplusu_j)"),
            (energy_refinement is not None, "Energy_Refinement"),
            (polarizability, "Polarizability"),
            (nscf, "a non-self-consistent calculation (nscf)"),
            (output_gw, "Output_GW"),
            (bands, "a band structure (bandstruc)"),
            # The restart file holds V, not vtau and W: the loop would start
            # without the orbital terms (and, in the orbital form of the
            # density gradient, without the gradient part of the potential).
            (restart, "Restart_Run (the restart file, parsec.dat, holds no vtau or W)"),
        )
        if condition
    ]
    if refused:
        raise UnsupportedParsecOptionError(
            f"Correlation_Type: {scf.xc_functional} (a meta-GGA) is not implemented together "
            "with " + ", ".join(refused)
        )


def _check_hybrid_request(
    scf,
    *,
    is_periodic: bool,
    periodicity: int = 0,
    specifications,
    energy_refinement,
    embedding: bool,
    polarizability: bool,
    nscf: bool,
    output_gw: bool,
    bands: bool,
) -> None:
    """Refuse by name what the hybrid functionals do not cover yet.

    The Fock loop is that of an isolated system with real orbitals on the
    full grid, or of a bulk cell or a slab (``periodicity`` 3 or 2) at the
    Gamma point or with a uniform k-point mesh (:mod:`parsec_python.SCF.hybrid_periodic`),
    spin-unpolarized or collinear spin.  A wire is refused, and so is
    ``Exx_Pair_Solver: cg`` on a periodic cell (its pair potentials are
    solved by FFT with the cell's exchange kernel).
    """

    hubbard = any(
        specification.hubbard_u != 0.0 or specification.hubbard_j != 0.0
        for specification in specifications.values()
    )
    refused = [
        label
        for condition, label in (
            (is_periodic and periodicity == 1, "Boundary_Conditions: wire"),
            (
                is_periodic and getattr(scf.exact_exchange, "pair_solver", "kronecker") != "kronecker",
                "Exx_Pair_Solver: cg on a periodic cell (its pair potentials are solved by FFT)",
            ),
            (
                not is_periodic
                and (
                    getattr(scf.exact_exchange, "divergence", None) is not None
                    or getattr(scf.exact_exchange, "auxiliary_exponent", None) is not None
                ),
                "Exx_Divergence or Exx_Auxiliary_Exponent on an isolated system (they belong to periodic cells)",
            ),
            (
                periodicity != 2 and getattr(scf.exact_exchange, "slab_coulomb", "padded") != "padded",
                "Exx_Slab_Coulomb outside a slab",
            ),
            (
                scf.self_consistent_spin_orbit
                or any(specification.spin_orbit for specification in specifications.values()),
                "spin-orbit coupling (SO_PSP, SO_from_scratch)",
            ),
            (scf.noncollinear, "non-collinear magnetism"),
            (embedding, "FDET (Potential_Field)"),
            (hubbard, "LDA+U (ldaplusu_u, ldaplusu_j)"),
            (energy_refinement is not None, "Energy_Refinement (the hybrid is already self-consistent)"),
            (scf.van_der_waals, "Van_Der_Waals"),
            (polarizability, "Polarizability"),
            (nscf, "a non-self-consistent calculation (nscf)"),
            (output_gw, "Output_GW"),
            (bands, "a band structure (bandstruc)"),
        )
        if condition
    ]
    if refused:
        raise UnsupportedParsecOptionError(
            f"Correlation_Type: {scf.xc_functional} (a self-consistent hybrid functional) is "
            "implemented for isolated systems, bulk cells and slabs, spin-unpolarized or collinear spin; not "
            "together with " + ", ".join(refused)
        )


def _check_libxc_request(scf, *, specifications, embedding: bool) -> None:
    """Refuse by name what the libxc functionals are not run with (untested combinations).

    Spin-orbit coupling and non-collinear magnetism (the local spin frame of
    ``V_xc.noncollinear``) and FDET (its own XC terms of the subsystems).
    """

    refused = [
        label
        for condition, label in (
            (
                scf.self_consistent_spin_orbit
                or any(specification.spin_orbit for specification in specifications.values()),
                "spin-orbit coupling (SO_PSP, SO_from_scratch)",
            ),
            (scf.noncollinear, "non-collinear magnetism"),
            (embedding, "FDET (Potential_Field)"),
        )
        if condition
    ]
    if refused:
        raise UnsupportedParsecOptionError(
            f"the libxc functional {scf.xc_functional} runs unpolarized or with collinear spin; not "
            "together with " + ", ".join(refused)
        )


#: The lower-bound rule of the later-SCF SUBSPACE filter of a periodic cell
#: (periodicity 1 wire, 2 slab, 3 bulk) whose input has no ``Smart_Subspace``
#: line: ``monotone`` is the earlier PARSEC's ``max(r_max, 0)``, which a bulk
#: cell's spectrum far below 0 Ry keeps at 0 for the whole SCF; ``smart`` is
#: PARSEC 1.5's ``Smart_Subspace: .true.`` (the bound at the largest saved Ritz
#: value, :func:`parsec_python.Eigensolvers.subspace.subspace_filter_bounds`).
#: Clusters always take the input's rule (``monotone`` without the line).  Bulk
#: cells take ``smart`` since 2026-11-08 (bulk diamond of
#: 512 atoms: 16 SCF iterations at the converged energy against 65 and 4.9
#: mRy/atom high); a slab keeps a level near 0 and was slower with ``smart``
#: (h-BN 30 iterations against 25, the same energy), wires were not measured.
DEFAULT_PERIODIC_SUBSPACE_LOWER_BOUND = {1: "monotone", 2: "monotone", 3: "smart"}

#: The short-Lanczos upper bound of the Chebyshev filters (CHEBFF, CHEBDAV,
#: SUBSPACE) of a periodic cell whose input has no ``Bounds_Method`` or
#: ``Bounds_Iter`` line: ``parsec`` is the earlier PARSEC's spectral radius
#: ``max(theta_max, |theta_min|)``, which for a bulk cell whose eigenvalues lie
#: far below 0 Ry (this code's periodic potential) takes ``|theta_min|`` and
#: widens the filter interval several times; ``lanczos`` is PARSEC 1.5's
#: ``Bounds_Method: lanczos`` (``theta_max``, ``max(degree/2, Bounds_Iter)``
#: steps; :func:`parsec_python.Eigensolvers.spectral_bounds.lanczos_bound_from_ritz`).
#: Clusters always take the input's bound (``parsec`` without the lines).
DEFAULT_PERIODIC_LANCZOS_BOUNDS = {1: "parsec", 2: "parsec", 3: "parsec"}

#: The preconditioner of ``Mixing_Preconditioner: auto`` (the default) for
#: ``Boundary_Conditions: bulk``; every other cell mixes without one.  ``none``
#: keeps the mixing of PARSEC; ``kerker`` here would make the Kerker filter the
#: default of bulk cells (Mixer/preconditioner.py), the combinations of
#: :func:`_check_mixing_preconditioner` excepted.
DEFAULT_BULK_MIXING_PRECONDITIONER = "none"


def _mixing_preconditioner_fields(scalar, one, *, periodicity: int, warnings: list) -> dict:
    """``Mixing_Preconditioner``, ``Kerker_Wavevector`` and ``Kerker_Dielectric_Constant`` (PARSEC.py).

    Fortran PARSEC has no preconditioner of its mixers.  ``auto`` (the default)
    is :data:`DEFAULT_BULK_MIXING_PRECONDITIONER` for a 3-D periodic cell and
    ``none`` for clusters, wires and slabs; ``kerker`` filters the residual of
    a bulk cell by ``P(G) = (G^2 + q0^2/eps)/(G^2 + q0^2)``, ``P(0) = 1``, with
    ``q0`` = ``Kerker_Wavevector`` (bohr^-1 by default, or ``ang^-1``) and
    ``eps`` = ``Kerker_Dielectric_Constant`` (default infinite: Kerker's
    ``G^2/(G^2 + q0^2)``).  Returns the :class:`MixingSettings` fields; at
    ``none`` the two numbers keep their defaults (a warning names them if given).
    """

    name = _normalize_label(one("mixing_preconditioner", "auto"))
    if name not in ("auto", *MIXING_PRECONDITIONERS):
        raise ParsecInputError(
            f"Mixing_Preconditioner: unknown preconditioner {name!r}; the choices are auto, "
            + ", ".join(MIXING_PRECONDITIONERS)
        )
    if name == "auto":
        name = DEFAULT_BULK_MIXING_PRECONDITIONER if periodicity == 3 else "none"
    given = [key for key in ("kerker_wavevector", "kerker_dielectric_constant") if key in scalar]
    if name != "kerker":
        for key in given:
            warnings.append(
                f"{'Kerker_Wavevector' if key == 'kerker_wavevector' else 'Kerker_Dielectric_Constant'} "
                "has no effect without Mixing_Preconditioner: kerker"
            )
        return {"preconditioner": name}
    fields = {"preconditioner": name}
    if "kerker_wavevector" in scalar:
        fields["kerker_wavevector"] = _inverse_length_bohr(
            one("kerker_wavevector"), label="Kerker_Wavevector"
        )
    if "kerker_dielectric_constant" in scalar:
        text = str(one("kerker_dielectric_constant")).strip().lower()
        fields["kerker_dielectric_constant"] = (
            float("inf")
            if text in ("inf", "infinity")
            else _strict_float(text, label="Kerker_Dielectric_Constant")
        )
    return fields


def _check_mixing_preconditioner(mixing, scf, *, periodicity: int, specifications, embedding: bool, requested: bool):
    """The mixing settings with the preconditioner the run can take.

    ``Mixing_Preconditioner: kerker`` is implemented for 3-D periodic cells,
    unpolarized or with collinear spin, at the Gamma point and on k-point
    meshes, with LDA/GGA functionals (native or libxc).  Where it is not (a
    cluster, wire or slab, a meta-GGA, a hybrid, spin-orbit coupling,
    non-collinear magnetism, FDET) a request is refused by name, and ``auto``
    falls back to ``none``.
    """

    if mixing.preconditioner == "none":
        return mixing
    refused = [
        label
        for condition, label in (
            (periodicity != 3, "a cluster, wire or slab (3-D periodic cells only)"),
            (scf.is_meta_gga, "a meta-GGA (its joined (V, vtau) mixing)"),
            (scf.is_hybrid, "a hybrid functional (the mixers of its Fock loop)"),
            (
                scf.self_consistent_spin_orbit
                or any(specification.spin_orbit for specification in specifications.values()),
                "spin-orbit coupling (SO_PSP, SO_from_scratch)",
            ),
            (scf.noncollinear, "non-collinear magnetism"),
            (embedding, "FDET (Potential_Field)"),
        )
        if condition
    ]
    if not refused:
        return mixing
    if requested:
        raise UnsupportedParsecOptionError(
            "Mixing_Preconditioner: kerker is not implemented together with " + ", ".join(refused)
        )
    return replace(mixing, preconditioner="none")


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", value.strip().lower())


def _strip_comment(line: str) -> str:
    positions = [line.find(marker) for marker in _COMMENT_MARKERS]
    positions = [position for position in positions if position >= 0]
    if positions:
        line = line[: min(positions)]
    return line.strip()


def _split_scalar(line: str, line_number: int) -> tuple[str, str]:
    match = re.match(r"^([A-Za-z0-9_.-]+)\s*(?::|=|\s)\s*(.*?)\s*$", line)
    if match is None:
        # ESDF permits a bare boolean label, interpreted as true.
        bare = line.strip()
        if re.fullmatch(r"[A-Za-z0-9_.-]+", bare):
            return bare, "true"
        raise ParsecInputError(f"line {line_number}: cannot parse input statement {line!r}")
    key, value = match.groups()
    return key, value or "true"


def _read_items(path: Path) -> list[_InputItem]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ParsecInputError(f"cannot read PARSEC input {path}: {error}") from error

    items: list[_InputItem] = []
    index = 0
    while index < len(lines):
        line_number = index + 1
        line = _strip_comment(lines[index])
        index += 1
        if not line:
            continue
        begin = re.match(r"^begin\s+(.+?)\s*$", line, flags=re.IGNORECASE)
        if begin is None:
            key, value = _split_scalar(line, line_number)
            items.append(_InputItem(_normalize_label(key), value, line_number))
            continue

        block_name = begin.group(1).strip()
        normalized_name = _normalize_label(block_name)
        block_lines: list[str] = []
        while index < len(lines):
            candidate_number = index + 1
            candidate = _strip_comment(lines[index])
            index += 1
            if not candidate:
                continue
            end = re.match(r"^end\s+(.+?)\s*$", candidate, flags=re.IGNORECASE)
            if end is not None:
                end_name = _normalize_label(end.group(1))
                if end_name != normalized_name:
                    raise ParsecInputError(
                        f"line {candidate_number}: block {block_name!r} "
                        f"closed by mismatched {end.group(1)!r}"
                    )
                break
            block_lines.append(candidate)
        else:
            raise ParsecInputError(
                f"line {line_number}: unterminated begin {block_name}"
            )
        items.append(
            _InputItem(
                normalized_name,
                tuple(block_lines),
                line_number,
                is_block=True,
            )
        )
    return items


def _fortran_float(value: str, *, label: str) -> float:
    token = value.strip().replace("D", "E").replace("d", "e")
    try:
        result = float(token)
    except ValueError as error:
        raise ParsecInputError(f"{label}: expected a number, got {value!r}") from error
    if not np.isfinite(result):
        raise ParsecInputError(f"{label}: value must be finite")
    return result


def _first_float(value: str, *, label: str) -> float:
    match = _FLOAT_PATTERN.search(value)
    if match is None:
        raise ParsecInputError(f"{label}: expected a numeric value")
    return _fortran_float(match.group(0), label=label)


def _strict_float(value: str, *, label: str) -> float:
    token = value.strip()
    if _FLOAT_PATTERN.fullmatch(token) is None:
        raise ParsecInputError(f"{label}: expected one number, got {value!r}")
    return _fortran_float(token, label=label)


def _integer(value: str, *, label: str) -> int:
    number = _strict_float(value, label=label)
    integer = int(number)
    if integer != number:
        raise ParsecInputError(f"{label}: expected an integer, got {value!r}")
    return integer


def _boolean(value: str, *, label: str) -> bool:
    normalized = re.sub(r"[.\s_-]", "", value.lower())
    if normalized in {"yes", "true", "t", "1", "on"}:
        return True
    if normalized in {"no", "false", "f", "0", "off"}:
        return False
    raise ParsecInputError(f"{label}: expected a boolean, got {value!r}")


def _physical_length(value: str, *, label: str) -> float:
    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[.\s_-]", "", remainder)
    if unit in {"", "bohr", "bohrs", "au", "atomicunit", "atomicunits"}:
        factor = 1.0
    elif unit in {"ang", "angstrom", "angstroms"}:
        factor = ANGSTROM_TO_BOHR
    else:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported length unit {remainder!r}; use bohr or ang"
        )
    return number * factor


def _energy_rydberg(value: str, *, label: str) -> float:
    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[.\s_-]", "", remainder)
    if unit in {"", "ry", "rydberg", "rydbergs"}:
        factor = 1.0
    elif unit in {"ha", "hartree", "hartrees"}:
        factor = 2.0
    elif unit in {"ev", "electronvolt", "electronvolts"}:
        factor = EV_TO_RYDBERG
    else:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported energy unit {remainder!r}; use Ry, Hartree, or eV"
        )
    return number * factor


def _esdf_energy_rydberg(value: str, *, label: str) -> float:
    """An energy in Ry with the conversion factors of ESDF (``esdf_physical``).

    ESDF's electronvolt is ``1.60219e-19/2.17991e-18`` Ry (1 Ry =
    13.60563 eV), not the CODATA value of :func:`_energy_rydberg`.
    """

    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[.\s_-]", "", remainder) or "ry"
    if unit not in _ESDF_ENERGY_JOULE:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported energy unit {remainder!r}; use Ry, eV or Hartree"
        )
    return number * _ESDF_ENERGY_JOULE[unit] / _ESDF_ENERGY_JOULE["ry"]


def _force_rydberg_per_bohr(value: str, *, label: str) -> float:
    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[.\s_-]", "", remainder)
    if unit in {"", "ry/bohr", "rydberg/bohr", "ry/au"}:
        factor = 1.0
    elif unit in {"ha/bohr", "hartree/bohr", "ha/au", "hartree/au"}:
        factor = 2.0
    elif unit in {"ev/ang", "ev/angstrom"}:
        factor = EV_TO_RYDBERG / ANGSTROM_TO_BOHR
    else:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported force unit {remainder!r}; use Ry/bohr, "
            "Hartree/bohr, or eV/ang"
        )
    return number * factor


def _force_ev_per_angstrom(value: str, *, label: str) -> float:
    """A force in eV/angstrom, the unit when the value names none."""

    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip()
    if not remainder:
        return _first_float(value, label=label)
    return _force_rydberg_per_bohr(value, label=label) * ANGSTROM_TO_BOHR / EV_TO_RYDBERG


def _unquoted(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def _mlff_settings(one, scalar, source: Path):
    """``MLFFSettings`` of ``Initial_Structure: mlff`` from the ``MLFF_*`` keywords.

    A model file, a ``python:file.py:function`` and ``MLFF_Python`` are
    taken relative to the input.  The interpreter path is made absolute but
    not resolved: a virtual environment's interpreter is a link.
    """

    import os
    import shutil

    from ..MLFF.settings import (
        DEFAULT_FMAX,
        DEFAULT_MODEL,
        DEFAULT_STEPS,
        DEFAULT_TASK,
        MLFFSettings,
    )

    def beside_input(text: str) -> str:
        path = Path(text).expanduser()
        return text if path.is_absolute() else os.path.abspath(source.parent / path)

    model = _unquoted(one("mlff_model", DEFAULT_MODEL))
    lower = model.lower()
    if lower.startswith("python:"):
        target = model.split(":", 1)[1]
        location, separator, function = target.rpartition(":")
        if separator and location.lower().endswith(".py"):
            model = f"python:{beside_input(location)}:{function}"
    elif lower.endswith((".model", ".pt")):
        model = beside_input(model)
    else:
        model = lower
    python = None
    if "mlff_python" in scalar:
        raw = _unquoted(one("mlff_python"))
        if not raw:
            raise ParsecInputError("MLFF_Python cannot be empty")
        if any(mark in raw for mark in ("/", "\\")):
            python = Path(beside_input(raw))
        else:
            found = shutil.which(raw)
            if found is None:
                raise ParsecInputError(f"MLFF_Python: {raw} is not on the PATH")
            python = Path(found)

    def optional_integer(key: str, label: str):
        return _integer(one(key), label=label) if key in scalar else None

    try:
        return MLFFSettings(
            model=model,
            task=_normalize_label(one("mlff_task", DEFAULT_TASK)),
            fmax=(
                _force_ev_per_angstrom(one("mlff_fmax"), label="MLFF_Fmax")
                if "mlff_fmax" in scalar
                else DEFAULT_FMAX
            ),
            steps=_integer(one("mlff_steps", str(DEFAULT_STEPS)), label="MLFF_Steps"),
            optimizer=_normalize_label(one("mlff_optimizer", "lbfgs")),
            device=_normalize_label(one("mlff_device", "auto")),
            python=python,
            charge=optional_integer("mlff_charge", "MLFF_Charge"),
            spin=optional_integer("mlff_spin", "MLFF_Spin"),
            reuse=_boolean(one("mlff_reuse", "true"), label="MLFF_Reuse"),
        )
    except ValueError as error:
        raise ParsecInputError(str(error)) from error


def _temperature_kelvin(value: str, *, label: str) -> float:
    """A temperature in K.

    ESDF's ``k`` is an energy (``esdf_mod.f90`` phy_n(54), 1.38066e-23 J),
    so ``esdf_physical('fermi_temp', 80, 'k')`` converts an energy unit too:
    ``Fermi_Temp: 0.05 eV`` is 580.226 K (PARSEC 1.5 test
    TierIV/3d_MgAl2_orthorhombic).  The energy units of :data:`_ESDF_ENERGY_JOULE`
    are taken with ESDF's factors.
    """

    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[.\s_-]", "", remainder)
    if unit in {"", "k", "kelvin"}:
        return number
    if unit in _ESDF_ENERGY_JOULE:
        return number * _ESDF_ENERGY_JOULE[unit] / _ESDF_BOLTZMANN_JOULE
    raise UnsupportedParsecOptionError(
        f"{label}: unsupported temperature unit {remainder!r}; use K (or an energy unit of "
        "ESDF: Ry, eV, Hartree, mRy, meV, mHartree)"
    )


def _charge_electrons(value: str, *, label: str) -> float:
    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[.\s_-]", "", remainder)
    if unit not in {"", "e", "electron", "electrons"}:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported charge unit {remainder!r}; use e"
        )
    return number


#: Electric-field units of ``esdf_mod.f90`` (``phy_u`` 48-53) in V/m, and
#: the aliases this reader also accepts.
ELECTRIC_FIELD_UNITS_V_PER_M = {
    "v/m": 1.0,
    "v/nm": 1.0e9,
    "v/ang": 1.0e10,
    "v/bohr": 1.8897268e10,
    "ry/bohr/e": 2.5711273e11,
    "har/bohr/e": 5.1422546e11,
    # Aliases of this reader, not of ESDF.
    "rydberg/bohr/e": 2.5711273e11,
    "ha/bohr/e": 5.1422546e11,
    "hartree/bohr/e": 5.1422546e11,
    "ev/ang/e": 1.0e10,
}


def _electric_field_value(value: str, *, label: str) -> float:
    """An electric field in ``ry/bohr/e``, the unit of ``usrinputfile.F90``.

    The third token names the unit as in ``esdf_physical``; ESDF drops case
    and the characters ``.`` and ``-``.  (The Fortran reads that token
    list-directed, which ends a word at a slash, so that a unit given there
    stops PARSEC with "Dimensions Do not match"; only the default unit
    works in the Fortran.  This reader converts the units as ESDF intends.)
    """

    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[\s.-]", "", remainder)
    if not unit:
        return number
    factor = ELECTRIC_FIELD_UNITS_V_PER_M.get(unit)
    if factor is None:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported electric-field unit {remainder!r}; use one of "
            "v/m, v/nm, v/ang, v/bohr, ry/bohr/e, har/bohr/e"
        )
    return number * factor / ELECTRIC_FIELD_UNITS_V_PER_M["ry/bohr/e"]


def _block_numbers(lines: tuple[str, ...], *, label: str) -> tuple[list[float], str]:
    values: list[float] = []
    unit_tokens: list[str] = []
    for line in lines:
        for token in re.split(r"[\s,]+", line.strip()):
            if not token:
                continue
            if _FLOAT_PATTERN.fullmatch(token):
                values.append(_fortran_float(token, label=label))
            else:
                unit_tokens.append(token)
    unit = " ".join(unit_tokens)
    return values, unit


def _parse_cell_shape(
    scalar: dict[str, list["_InputItem"]],
    blocks: dict[str, list["_InputItem"]],
    *,
    slab: bool = False,
) -> PeriodicCell:
    """Parse a ``Cell_Shape`` block into a :class:`PeriodicCell`.

    A bulk cell takes one line (orthorhombic side lengths) or three lines (lattice vectors).  A slab takes the two
    lattice vectors of the xy plane (two or three components each) and the height ``2R`` of its box along z from
    ``Boundary_Sphere_Radius``.
    """

    label = "Cell_Shape"
    cell_blocks = blocks.get("cell_shape", [])
    if len(cell_blocks) != 1:
        raise ParsecInputError("a periodic cell requires exactly one Cell_Shape block")
    lines = cell_blocks[0].value
    if slab:
        if len(lines) != 2:
            raise ParsecInputError(f"a slab Cell_Shape block has 2 lines (the in-plane lattice vectors), not {len(lines)}")
        rows = []
        for row_index, line in enumerate(lines, start=1):
            row_values, unit = _block_numbers((line,), label=label)
            if unit or len(row_values) not in (2, 3) or (len(row_values) == 3 and row_values[2] != 0.0):
                raise ParsecInputError(
                    f"slab Cell_Shape line {row_index} must be an in-plane vector: two components, or three with z = 0"
                )
            rows.append([row_values[0], row_values[1]])
        radius_occurrences = scalar.get("boundary_sphere_radius", [])
        if len(radius_occurrences) != 1:
            raise ParsecInputError("a slab needs Boundary_Sphere_Radius, the half height of its box along z")
        height = 2.0 * _physical_length(str(radius_occurrences[0].value), label="Boundary_Sphere_Radius")
        vectors = np.zeros((3, 3))
        vectors[:2, :2] = np.array(rows, dtype=np.float64)
    elif len(lines) == 1:
        values, unit = _block_numbers(lines, label=label)
        if len(values) != 3 or unit:
            raise ParsecInputError(
                "a one-line Cell_Shape block must contain exactly the three "
                "orthorhombic side lengths"
            )
        vectors = np.diag(values)
    elif len(lines) == 3:
        rows = []
        for row_index, line in enumerate(lines, start=1):
            row_values, unit = _block_numbers((line,), label=label)
            if len(row_values) != 3 or unit:
                raise ParsecInputError(
                    f"Cell_Shape line {row_index} must contain exactly three components"
                )
            rows.append(row_values)
        vectors = np.array(rows, dtype=np.float64)
    else:
        raise ParsecInputError(
            "Cell_Shape must contain 1 line (orthorhombic side lengths) or "
            f"3 lines (general lattice vectors), not {len(lines)}"
        )

    scale_occurrences = scalar.get("lattice_vector_scale", [])
    if len(scale_occurrences) > 1:
        lines_str = ", ".join(str(item.line) for item in scale_occurrences)
        raise ParsecInputError(
            f"duplicate lattice_vector_scale values on lines {lines_str}; provide it once"
        )
    scale_text = (
        str(scale_occurrences[0].value) if scale_occurrences else "1 bohr"
    )
    scale = _physical_length(scale_text, label="Lattice_Vector_Scale")
    if slab:
        vectors = vectors * scale
        vectors[2, 2] = height  # Boundary_Sphere_Radius is a length of its own, not scaled
        return PeriodicCell(lattice_vectors=vectors, dimensions=2)
    return PeriodicCell(lattice_vectors=vectors * scale)


def _lattice_vector_scale(scalar: dict[str, list["_InputItem"]]) -> float:
    scale_occurrences = scalar.get("lattice_vector_scale", [])
    if len(scale_occurrences) > 1:
        lines_str = ", ".join(str(item.line) for item in scale_occurrences)
        raise ParsecInputError(
            f"duplicate lattice_vector_scale values on lines {lines_str}; provide it once"
        )
    scale_text = str(scale_occurrences[0].value) if scale_occurrences else "1 bohr"
    return _physical_length(scale_text, label="Lattice_Vector_Scale")


def _parse_wire_slab_cell(
    scalar: dict[str, list["_InputItem"]],
    blocks: dict[str, list["_InputItem"]],
    periodicity: int,
) -> PeriodicCell:
    """``Cell_Shape`` of a wire or slab (``usrinputfile.F90:460-526``).

    A wire has one line with its period ``a`` (``latt_vec(1,1)``, along
    ``x``); a slab two lines with ``a1`` and ``a2`` in the ``xy`` plane.  The
    confined directions get unit vectors (``usrinputfile.F90:777-779``), and
    ``Lattice_Vector_Scale`` multiplies the vectors.  Fortran reads three
    numbers per slab line (a line of two is error 107, although the guide
    shows two); here two numbers or three with a zero third are accepted.
    Fortran reads only the first number of a wire line; further numbers
    other than zero are refused here.
    """

    label = "Cell_Shape"
    cell_blocks = blocks.get("cell_shape", [])
    if len(cell_blocks) != 1:
        # usrinputfile.F90 error 108.
        raise ParsecInputError(
            "a wire or slab requires exactly one Cell_Shape block"
        )
    lines = cell_blocks[0].value
    if len(lines) != periodicity:
        # usrinputfile.F90 error 106.
        raise ParsecInputError(
            f"the Cell_Shape block of a {'wire' if periodicity == 1 else 'slab'} "
            f"has {periodicity} line(s), not {len(lines)}"
        )
    rows = []
    for row_index, line in enumerate(lines, start=1):
        values, unit = _block_numbers((line,), label=label)
        if unit or not values:
            raise ParsecInputError(f"Cell_Shape line {row_index} is not a list of numbers")
        if periodicity == 1:
            if any(value != 0.0 for value in values[1:]):
                raise ParsecInputError(
                    "the Cell_Shape of a wire is its period along x, one number "
                    "(PARSEC reads only the first)"
                )
            if not values[0] > 0.0:
                raise ParsecInputError("the period of a wire must be positive")
            rows.append([values[0], 0.0, 0.0])
            continue
        if len(values) == 2:
            values = [values[0], values[1], 0.0]
        if len(values) != 3:
            raise ParsecInputError(
                f"Cell_Shape line {row_index} of a slab needs the x and y "
                "components of a lattice vector (and a zero z component)"
            )
        if values[2] != 0.0:
            raise ParsecInputError(
                "the lattice vectors of a slab lie in the xy plane: the third "
                f"component of Cell_Shape line {row_index} must be zero"
            )
        rows.append(list(values))
    from ..Grid.confined import confined_lattice_vectors

    vectors = confined_lattice_vectors(np.asarray(rows, dtype=np.float64) * _lattice_vector_scale(scalar))
    try:
        return PeriodicCell(lattice_vectors=vectors)
    except ValueError as error:
        raise ParsecInputError(f"Cell_Shape: {error}") from error


def _charged_sheets(
    scalar: dict[str, list["_InputItem"]],
    blocks: dict[str, list["_InputItem"]],
    requested: bool,
    periodicity: int,
    coordinate_matrix: np.ndarray | None,
    coordinate_factor: float,
    warnings: list[str],
) -> tuple[ChargedSheet, ...]:
    """The charged sheets of a slab (``usrinputfile.F90:2222-2299``).

    ``add_charged_sheets`` (false), ``num_charged_sheets`` types (error 146
    without it), and for every type the next ``sheet_charge`` (e, ESDF's
    cursor) and the next ``Sheet_Coord`` block (error 148 without it), one
    sheet per line; the coordinates take ``Coordinate_Unit`` and only ``z``
    is used.  Charged sheets need a slab (error 145).
    """

    keys = ("num_charged_sheets", "sheet_charge")
    if not requested:
        if any(key in scalar for key in keys) or "sheet_coord" in blocks:
            warnings.append(
                "num_charged_sheets, sheet_charge and Sheet_Coord are ignored "
                "without add_charged_sheets: .true., as in PARSEC"
            )
        return ()
    if periodicity != 2:
        raise ParsecInputError(
            "support for charged sheets is only available in 2-D (slab) systems "
            "(PARSEC error 145)"
        )
    counts = scalar.get("num_charged_sheets", [])
    if not counts:
        raise ParsecInputError(
            "unknown number of charged sheets: num_charged_sheets is missing "
            "(PARSEC error 146)"
        )
    if len(counts) > 1:
        raise ParsecInputError("provide num_charged_sheets once")
    types = _integer(str(counts[0].value), label="num_charged_sheets")
    if types < 1:
        raise ParsecInputError("num_charged_sheets must be at least 1")
    charges = scalar.get("sheet_charge", [])
    coordinate_blocks = blocks.get("sheet_coord", [])
    if len(charges) < types:
        raise ParsecInputError(
            f"num_charged_sheets is {types}, but {len(charges)} sheet_charge "
            "line(s) were found (PARSEC reads one per type)"
        )
    if len(coordinate_blocks) < types:
        raise ParsecInputError(
            f"no charged sheet coordinates were found for sheet type "
            f"{len(coordinate_blocks) + 1} (PARSEC error 148)"
        )
    if len(charges) > types or len(coordinate_blocks) > types:
        warnings.append(
            "more sheet_charge lines or Sheet_Coord blocks than num_charged_sheets: "
            "PARSEC reads the first ones only, and so does this run"
        )
    sheets = []
    for index in range(types):
        charge_values, unit = _block_numbers((str(charges[index].value),), label="sheet_charge")
        if len(charge_values) != 1 or unit.strip().lower() not in {"", "e"}:
            raise ParsecInputError("sheet_charge must be a charge in e")
        block = coordinate_blocks[index]
        if not block.value:
            raise ParsecInputError(
                f"Sheet_Coord block {index + 1} holds no coordinates (PARSEC error 148)"
            )
        for line in block.value:
            values, unit = _block_numbers((line,), label="Sheet_Coord")
            if unit or len(values) < 3:
                raise ParsecInputError("every Sheet_Coord line needs x y z")
            if coordinate_matrix is None:
                position = np.asarray(values[:3], dtype=float) * coordinate_factor
            else:
                position = np.asarray(values[:3], dtype=float) @ coordinate_matrix
            sheets.append(ChargedSheet(charge=charge_values[0], height=float(position[2])))
    return tuple(sheets)


def _grid_shift_block(
    blocks: dict[str, list["_InputItem"]], notes: list[str]
) -> tuple[float, float, float] | None:
    """The ``Grid_Shift`` block of PARSEC 1.5 (``io/usrinputfile.F90:1369-1382``), ``None`` without it.

    One line of three numbers, the offset of the first grid point along each lattice direction in grid steps;
    Fortran reads the first line only (with a warning for more) and takes the absolute values.  The caller
    zeroes the shift along an oblique periodic direction (:func:`parsec_python.Grid.pbc.periodic_grid_shift`).
    With the block the shift is the input's whatever ``Ignore_Symmetry`` says (the previous Fortran tree, and
    this port before the parity round of 1.5, had no block and forced zero with ``Ignore_Symmetry``).  Values of
    a whole step or more are refused: they move the points of a periodic grid out of its cell and the cube of
    a cluster grid off its sphere (``n_i + shift_i`` is the same grid as ``shift_i`` modulo whole steps).
    """

    found = blocks.get("grid_shift", [])
    if not found:
        return None
    if len(found) != 1:
        raise ParsecInputError("give one Grid_Shift block")
    lines = tuple(found[0].value)
    if not lines:
        raise ParsecInputError("the Grid_Shift block is empty: give one line of three numbers")
    if len(lines) != 1:
        notes.append(
            "Grid_Shift only needs one line: only the first line of the block is taken, as in PARSEC 1.5"
        )
    values, unit = _block_numbers(lines[:1], label="Grid_Shift")
    if unit or len(values) < 3:
        raise ParsecInputError(
            "Grid_Shift needs one line of three numbers: the shift of the grid along each lattice "
            "direction, in grid steps"
        )
    shift = tuple(abs(float(value)) for value in values[:3])
    if any(value >= 1.0 for value in shift):
        raise UnsupportedParsecOptionError(
            f"Grid_Shift {' '.join(f'{value:g}' for value in shift)}: give the shift in [0, 1) grid steps "
            "(a whole step more is the same grid, numbered differently)"
        )
    notes.append(
        "Grid Shift is read from input (PARSEC 1.5: 'This has untested consequences!'): "
        + " ".join(f"{value:g}" for value in shift)
        + " grid steps"
    )
    return shift


def _foldblock_side_length(scalar, one, *, periodicity: int, grid_notes: list[str]) -> int:
    """``Foldblock_Side_Length`` of PARSEC 1.5 (``io/usrinputfile.F90:798-817``, ``grid/grid_partition.f90:525-607``).

    PARSEC 1.5 partitions the grid into blocks of ``s**3`` points; a block whose any point lies inside the domain
    enters it whole, and ``s != 1`` switches the symmetry off (the caller does).  Here:

    * ``1`` (the default): nothing changes;
    * ``2`` for a cluster: its domain filled up to whole blocks of 2 x 2 x 2 points, the blocks starting at even
      indices as Fortran's (its padded box ``bbox_pad`` starts at an even index, ``grid_partition.f90:258-320``;
      :func:`parsec_python.Grid.cluster.build_cluster_grid`);
    * ``2`` for a bulk cell with an even half order ``Expansion_Order/2``: the blocks fill the cell exactly
      (Fortran's padded box starts ``Expansion_Order/2`` points below ``-n/2``, ``n`` even), so the grid is the one
      of ``1``.

    Refused: values below 1 (PARSEC 1.5 then takes blocks of 1 x 2 x 4), other sides (their alignment with the box
    and, for a bulk cell, ``n = s nint(L/(s h))`` of ``inipbc.f90:174-249`` are not implemented), a wire or slab
    (blocks along its confined directions), a bulk cell with an odd half order (Fortran's blocks then start half a
    block off the cell and its periodic domain takes a layer of points more: "PBC in folds is incomplete").
    """

    if "foldblock_side_length" not in scalar:
        return 1
    value = _integer(one("foldblock_side_length"), label="Foldblock_Side_Length")
    if value == 1:
        return 1
    if value < 1:
        raise UnsupportedParsecOptionError(
            f"Foldblock_Side_Length {value}: give a positive side (PARSEC 1.5 replaces a smaller one by "
            "blocks of 1 x 2 x 4 points, which is not implemented)"
        )
    if value != 2:
        raise UnsupportedParsecOptionError(
            f"Foldblock_Side_Length {value}: only blocks of side 1 and 2 are implemented"
        )
    if periodicity in (1, 2):
        raise UnsupportedParsecOptionError(
            "Foldblock_Side_Length 2 with a wire or slab: PARSEC 1.5 fills its confined directions with "
            "whole blocks, which is not implemented"
        )
    if periodicity == 3:
        half_order = _integer(one("expansion_order", "12"), label="Expansion_Order") // 2
        if half_order % 2:
            raise UnsupportedParsecOptionError(
                "Foldblock_Side_Length 2 with an odd Expansion_Order/2: the blocks of PARSEC 1.5 then start "
                "half a block off the cell, and its periodic domain has a layer of points more ('PBC in folds "
                "is incomplete')"
            )
        grid_notes.append(
            "Foldblock_Side_Length 2: the blocks of 2 x 2 x 2 points fill the periodic cell, so the grid is "
            "the one of blocks of 1"
        )
    return value


def _fortran_bulk_box_lengths(grid: PeriodicGridSettings, periodic_cell: PeriodicCell, general: bool):
    """``box_lengths`` of the bulk grid of the Fortran-derived routines: the lengths of the lattice vectors on
    Fortran's general grid, the diagonal of the cell on the orthorhombic one (the values and types each grid
    has always printed: the settings' text is a cache key)."""

    vectors = periodic_cell.lattice_vectors
    if general:
        lengths = tuple(float(value) for value in np.linalg.norm(vectors, axis=1))
    else:
        lengths = tuple(np.diag(vectors))
    return replace(grid, box_lengths=lengths)


def _parse_super_cell_block(
    blocks: dict[str, list["_InputItem"]],
    key: str,
    *,
    minimum: int,
    periodic: bool,
    count: int = 3,
) -> tuple[int, int, int] | None:
    """One ``Super_Cell``/``Super_Cell_Vac`` block (F ``usrinputfile.F90:723-773``).

    One line (errors 158/160) whose first ``count`` entries are integers
    (159/161); later entries are ignored, as by Fortran's list-directed
    read.  Fortran does not check the values; here a replication below 1 or
    a negative vacuum factor, which would give a degenerate or reversed
    cell, is refused.  A cluster reads nothing from the line, so there only
    the line count is checked.  A wire or slab reads ``count = pbc%per``
    entries (``supercell(1:pbc%per)``); the confined directions keep one
    copy and no vacuum.
    """

    label = "Super_Cell" if key == "super_cell" else "Super_Cell_Vac"
    items = blocks.get(key, [])
    if not items:
        return None
    if len(items) > 1:
        raise ParsecInputError(f"provide a single {label} block")
    lines = items[0].value
    if len(lines) != 1:
        raise ParsecInputError(
            f"there should be 1 line instead of {len(lines)} in the {label} block"
        )
    if not periodic:
        return None
    tokens = [token for token in re.split(r"[\s,]+", lines[0].strip()) if token]
    if len(tokens) < count or not all(
        re.fullmatch(r"[+-]?\d+", token) for token in tokens[:count]
    ):
        raise ParsecInputError(
            f"bad specification of the {label} block: give "
            + ("three integers" if count == 3 else f"{count} integer(s)")
            + f", one per periodic lattice vector, got {lines[0].strip()!r}"
        )
    values = tuple(int(token) for token in tokens[:count])
    if any(value < minimum for value in values):
        raise ParsecInputError(
            f"{label} values must be at least {minimum}, got {list(values)}"
        )
    return values + ((1,) if minimum == 1 else (0,)) * (3 - count)


def _coordinate_factor(value: str) -> float:
    normalized = _normalize_label(value)
    if normalized in {"cartesian_ang", "cartesianang"}:
        return ANGSTROM_TO_BOHR
    if normalized in {
        "cartesian_bohr",
        "cartesianbohr",
        "cartesian_au",
        "cartesianau",
    }:
        return 1.0
    raise UnsupportedParsecOptionError(
        "Coordinate_Unit must be Cartesian_Ang, Cartesian_Bohr or (for a "
        "periodic system) Lattice_Vectors"
    )


def _lattice_is_diagonal(lattice_vectors: np.ndarray) -> bool:
    return bool(np.allclose(lattice_vectors, np.diag(np.diag(lattice_vectors)), atol=1.0e-10))


def _is_lattice_coordinate_unit(value: str) -> bool:
    """``Coordinate_Unit: lattice_vectors``: coordinates in units of the lattice vectors (fractional)."""
    return _normalize_label(value) in {"lattice_vectors", "latticevectors"}


#: The formats of the species that read a per-species label in
#: ``usrinputfile.F90:1806-1867``.  ESDF keeps one cursor per label
#: (``esdf_mod.f90``), so the n-th occurrence of such a label goes to the n-th
#: species that reads it, whatever the species in between.  Every other
#: per-species label is read by every species.
FORMATS_READING = {
    "local_component": frozenset({"martins_new", "martins_old", "fhipp"}),
    # PARSEC 1.5 reads Potential_Num for martins_old and fhipp only
    # (1.5's usrinputfile.F90:3244-3248); the tree before it read it for every species.
    "potential_num": frozenset({"martins_old", "fhipp"}),
    "core_cutoff_radius": frozenset({"martins_old", "martins_wang", "fhipp"}),
    "electron_per_orbital": frozenset({"martins_old", "fhipp"}),
    "nonlinear_core_correction": frozenset({"fhipp"}),
}

#: File name of each format: ``<Atom_Type>_<suffix>.DAT`` (``pseudo.f90:143-152``).
POTENTIAL_FILE_SUFFIX = {
    "martins_new": "POTRE",
    "martins_old": "POTR",
    "martins_wang": "POTRW",
    "fhipp": "FHIPP",
}


def _format_input(
    symbol: str,
    pseudo_format: str,
    scalar: dict[str, list["_InputItem"]],
    blocks: dict[str, list["_InputItem"]],
    species_index: int,
    cursors: dict[str, int],
) -> dict[str, object]:
    """Read what a ``martins_old``, ``martins_wang`` or ``fhipp`` species adds.

    PARSEC 1.5's ``usrinputfile.F90`` (3244-3330) reads ``Potential_Num``
    for the old and FHIPP formats, ``Core_Cutoff_Radius`` for the old,
    Martins-Wang and FHIPP formats, ``Electron_Per_Orbital`` for the old and
    FHIPP formats and ``Nonlinear_Core_Correction`` for FHIPP only, and ESDF
    hands out the occurrences of a label in the order of the calls.  A
    species therefore takes each of them at its position among the species
    that read it: ``cursors[label]``, which is advanced here.  (The tree
    before 1.5 read ``Potential_Num`` for every species, so that a species
    took the one of its position among all species; ``species_index`` is no
    longer used for it.)

    A Martins-Wang species takes everything but ``Core_Cutoff_Radius`` from
    its file and reads no ``Potential_Num``.

    Where PARSEC would go on with a default (no nonlocal projectors without
    a cutoff radius, no ionic charge or no occupations without the block)
    this reader stops.
    """

    def take(label: str) -> int:
        position = cursors.get(label, 0)
        cursors[label] = position + 1
        return position

    if pseudo_format == "martins_wang":
        return {
            "pseudopotential_format": pseudo_format,
            "core_cutoff_radius": _core_cutoff_radius(
                symbol, pseudo_format, scalar, take("core_cutoff_radius")
            ),
        }
    counts = scalar.get("potential_num", [])
    count_index = take("potential_num")
    if count_index >= len(counts):
        raise ParsecInputError(
            f"Atom_Type {symbol}: a {pseudo_format} pseudopotential needs "
            "Potential_Num.  PARSEC 1.5 gives the n-th Potential_Num of the input "
            "to the n-th Atom_Type of format martins_old or fhipp: "
            f"{len(counts)} were found and {symbol} is number "
            f"{count_index + 1} of these species"
        )
    channels = _integer(
        str(counts[count_index].value), label=f"{symbol} Potential_Num"
    )
    if not 1 <= channels <= 4:
        raise ParsecInputError(
            f"Atom_Type {symbol}: impossible number of potentials "
            f"{channels}; Potential_Num must be 1 to 4"
        )
    radius = _core_cutoff_radius(
        symbol, pseudo_format, scalar, take("core_cutoff_radius")
    )
    orbital_index = take("electron_per_orbital")
    orbital_blocks = blocks.get("electron_per_orbital", [])
    if orbital_index >= len(orbital_blocks):
        raise ParsecInputError(
            f"Atom_Type {symbol}: a {pseudo_format} pseudopotential needs an "
            + (
                "Electron_Per_Orbital block; its sum is the ionic charge"
                if pseudo_format == "martins_old"
                else "Electron_Per_Orbital block, the occupations of its channels"
            )
        )
    lines = orbital_blocks[orbital_index].value
    if len(lines) != 1:
        raise ParsecInputError(
            f"Atom_Type {symbol}: there should be only one line in the "
            "Electron_Per_Orbital block"
        )
    electrons, unit = _block_numbers(lines, label=f"{symbol} Electron_Per_Orbital")
    if unit or len(electrons) < channels:
        raise ParsecInputError(
            f"Atom_Type {symbol}: Electron_Per_Orbital must hold the "
            f"occupations of the {channels} channels of Potential_Num"
        )
    result: dict[str, object] = {
        "pseudopotential_format": pseudo_format,
        "number_of_channels": channels,
        # PARSEC reads Potential_Num values and ignores the rest of the line.
        "electrons_per_orbital": tuple(electrons[:channels]),
        "core_cutoff_radius": radius,
    }
    if pseudo_format == "fhipp":
        flags = scalar.get("nonlinear_core_correction", [])
        position = take("nonlinear_core_correction")
        result["nonlinear_core_correction"] = (
            _boolean(
                str(flags[position].value),
                label=f"{symbol} Nonlinear_Core_Correction",
            )
            if position < len(flags)
            else False
        )
    return result


def _core_cutoff_radius(
    symbol: str,
    pseudo_format: str,
    scalar: dict[str, list["_InputItem"]],
    index: int,
) -> float:
    radii = scalar.get("core_cutoff_radius", [])
    if index >= len(radii):
        raise ParsecInputError(
            f"Atom_Type {symbol}: a {pseudo_format} pseudopotential needs "
            "Core_Cutoff_Radius, the radius of its nonlocal projectors"
        )
    radius = _physical_length(
        str(radii[index].value), label=f"{symbol} Core_Cutoff_Radius"
    )
    if not radius > 0.0:
        raise ParsecInputError(
            f"Atom_Type {symbol}: Core_Cutoff_Radius must be positive"
        )
    return radius


def _has_d_channel(path: Path, format_input: dict[str, object]) -> bool:
    """Whether the pseudopotential file holds a d pseudo-wavefunction."""

    if format_input:
        return int(format_input["number_of_channels"]) > 2
    from ..Pseudopotential import read_parsec_pseudopotential

    return 2 in read_parsec_pseudopotential(path).radial_wavefunctions


def _find_potential(
    symbol: str,
    input_directory: Path,
    pseudopotential_directory: Path | None,
    extension: str = "POTRE",
) -> Path:
    filename = f"{symbol}_{extension}.DAT"
    directories: list[Path] = []
    if pseudopotential_directory is not None:
        directories.append(pseudopotential_directory)
    directories.append(input_directory)
    current = Path.cwd().resolve()
    if current not in directories:
        directories.append(current)

    tried: list[Path] = []
    for directory in directories:
        candidate = (directory / filename).resolve()
        tried.append(candidate)
        if candidate.is_file():
            return candidate
    attempted = "\n  ".join(str(path) for path in tried)
    raise ParsecInputError(
        f"missing pseudopotential for Atom_Type {symbol!r}; searched:\n  {attempted}"
    )


# --- PARSEC 1.5 keywords of the input layer (see _INPUT15_* above) ----------


def _apply_input15_aliases(scalar: dict[str, list["_InputItem"]]) -> None:
    """Read the PARSEC 1.5 name of a renamed keyword as the old one (:data:`_INPUT15_ALIASES`).

    ``Mol_RNG_Seed`` (``usrinputfile.F90:4741``) is the ``RNG_Seed`` of the
    tree before 1.5.  An input that gives both names is refused.
    """

    for new, old in _INPUT15_ALIASES.items():
        if new not in scalar:
            continue
        if old in scalar:
            lines = ", ".join(str(item.line) for item in (*scalar[new], *scalar[old]))
            raise ParsecInputError(
                f"{new} is PARSEC 1.5's name of {old}: both are given (lines {lines}); "
                "provide one of them"
            )
        scalar[old] = [replace(item, key=old) for item in scalar.pop(new)]


def _input15_bak_constructs(
    scalar: dict[str, list["_InputItem"]],
    blocks: dict[str, list["_InputItem"]],
) -> list[str]:
    """Read what PARSEC 1.5 writes into its own bak_complete_parsec.in as 1.5 reads it.

    1.5's ``write_out_parameters`` (``usrinputfile.F90:5477-6042``) writes every block, empty where the run has
    none, ``Kpoint_Unit`` whatever ``Kpoint_Method`` says, and the format-dependent per-species labels
    (``Potential_Num``, ``Core_Cutoff_Radius``, ``Electron_Per_Orbital``, ``Nonlinear_Core_Correction``,
    ``Local_Component``) for every species.  How 1.5 reads them back:

    * an empty block: ``bandstruc`` and ``nscf_kpoints`` are skipped for ``nlines = 0`` (1777-1830); the blocks
      1.5 does not read for the run (``Kpoint_List`` without ``Kpoint_Method manual``, ``Monkhorst_Pack_Grid`` and
      ``_Shift`` without ``mp``, ``dos_mp_grid``/``_shift`` without ``Create_DOS``, ``nscf_kgrid``/``_shift``
      without ``Output_GW``, ``Initial_NCL_Moment`` without non-collinear magnetism, ``Cell_Shape`` of a cluster)
      are never looked at.  They are dropped here, as absent.  An empty ``Super_Cell`` or ``Super_Cell_Vac``
      stops 1.5 itself (ierr 158/160, also for a cluster: 1.5 does not read its own bak file of a cluster back,
      checked with the H2 run); it holds no replication and is read as absent, with the warning saying so.
      Every other empty block keeps its meaning (an empty ``Save_Wfn_Bands`` saves no state; an empty block 1.5
      reads for the run, such as ``Monkhorst_Pack_Grid`` with ``mp``, stays refused).
    * ``Kpoint_Unit`` (1618) is read always and used for a manual list only (1747): dropped without
      ``Kpoint_Method manual``.
    * ``Cluster_Domain_Shape undef``, 1.5's default value (902), is its sphere (1232-1247).
    * a format-dependent label is read for the formats of :data:`FORMATS_READING` only, the n-th occurrence by
      the n-th species that reads it (ESDF's cursor).  Where the input holds one occurrence under every
      Atom_Type and the species that read the label come first, every reader takes its own line and the lines of
      the others are never reached: they are dropped.  Otherwise 1.5 would hand a reader the line of another
      species (its own bak file of a martins_new species before a martins_old one gives the old species
      ``Potential_Num 0``, ierr 138): the input is left to the checks of the species loop, which refuse it.

    Returns the notes for the warnings.  Inputs without these constructs are not touched.
    """

    notes: list[str] = []

    def raw(key: str) -> str | None:
        items = scalar.get(key, [])
        return str(items[0].value) if items else None

    def flag(key: str) -> bool:
        text = raw(key)
        if text is None:
            return False
        try:
            return _boolean(text, label=key)
        except ParsecInputError:
            return False

    method = _normalize_label(raw("kpoint_method") or "none")
    periodic = _BOUNDARY_PERIODICITY.get(_normalize_label(raw("boundary_conditions") or "cluster"), 0) > 0
    unread_when_empty = {
        "bandstruc": True,
        "nscf_kpoints": True,
        "super_cell": True,
        "super_cell_vac": True,
        "kpoint_list": method != "manual",
        "monkhorst_pack_grid": method != "mp",
        "monkhorst_pack_shift": method != "mp",
        "dos_mp_grid": not flag("create_dos"),
        "dos_mp_shift": not flag("create_dos"),
        "nscf_kgrid": not flag("output_gw"),
        "nscf_kgrid_shift": not flag("output_gw"),
        "initial_ncl_moment": not flag("non_collinear_magnetism"),
        "cell_shape": not periodic,
    }
    dropped = []
    for key, unread in unread_when_empty.items():
        items = blocks.get(key, [])
        empty = [item for item in items if not item.value]
        if not unread or not empty or len(empty) != len(items):
            continue
        del blocks[key]
        dropped.append(key)
    if dropped:
        notes.append(
            "empty block(s) "
            + ", ".join(dropped)
            + " (as PARSEC 1.5 writes them into its bak_complete_parsec.in) are read as absent, as 1.5 does"
            + (
                " (an empty super_cell or super_cell_vac stops 1.5 itself, ierr 158/160)"
                if {"super_cell", "super_cell_vac"} & set(dropped)
                else ""
            )
        )
    if "kpoint_unit" in scalar and method != "manual":
        del scalar["kpoint_unit"]
        notes.append(
            "Kpoint_Unit is used for a Kpoint_List only (Kpoint_Method manual); PARSEC 1.5 reads it and ignores it "
            "here, and so does this run"
        )
    shape_items = scalar.get("cluster_domain_shape", [])
    if len(shape_items) == 1 and _normalize_label(str(shape_items[0].value)) == "undef":
        # 1.5's default value (usrinputfile.F90:902), which its bak file writes; any shape 1.5 does not know is
        # its sphere with a warning (1232-1247).  Other unknown names stay refused here (a misspelt shape).
        scalar["cluster_domain_shape"] = [replace(shape_items[0], value="sphere")]
        notes.append(
            "Cluster_Domain_Shape: undef is PARSEC 1.5's default (its bak_complete_parsec.in writes it), the sphere, "
            "as 1.5 reads it"
        )

    # The format-dependent per-species labels of 1.5's bak file (one line under every Atom_Type).
    type_lines = [item.line for item in scalar.get("atom_type", [])]
    if not type_lines:
        return notes
    formats_given = scalar.get("pseudopotential_format", [])
    formats = [
        _normalize_label(str(formats_given[index].value)) if index < len(formats_given) else "martins_new"
        for index in range(len(type_lines))
    ]
    formats = ["martins_new" if name == "martinsnew" else name for name in formats]

    def section(line: int) -> int:
        return sum(1 for start in type_lines if start < line) - 1

    unread_lines = []
    for label, readers in FORMATS_READING.items():
        table = blocks if label == "electron_per_orbital" else scalar
        occurrences = table.get(label, [])
        reading = [index for index, name in enumerate(formats) if name in readers]
        if len(occurrences) <= len(reading):
            continue
        if [section(item.line) for item in occurrences] != list(range(len(type_lines))):
            continue  # not one line under every Atom_Type: the species loop decides
        if reading != list(range(len(reading))):
            continue  # a reader after a species that does not read it: 1.5 would take another species' line
        table[label] = [occurrences[index] for index in reading]
        if not table[label]:
            del table[label]
        unread_lines.append(label)
    if unread_lines:
        notes.append(
            ", ".join(unread_lines)
            + ": one line under every Atom_Type, as PARSEC 1.5's bak_complete_parsec.in writes them; 1.5 reads "
            "them for some pseudopotential formats only, and the lines of the other species are ignored, as there"
        )
    return notes


def _species_potential_file(
    value: str,
    symbol: str,
    input_directory: Path,
    pseudopotential_directory: Path | None,
) -> Path:
    """The file of a ``Pseudopot_File`` line (PARSEC 1.5, ``usrinputfile.F90:3164-3188``).

    PARSEC 1.5 opens the path as given, from its run directory
    (``pseudo.f90``: ``fname = trim(clust%ppfile)``, cut to 90 characters,
    a defect this reader does not copy).  A relative path is looked for
    beside the input first, then in ``--pp-dir`` and the current directory;
    an absolute path is taken as it is.
    """

    raw = value.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1].strip()
    if not raw:
        raise ParsecInputError(f"Atom_Type {symbol}: Pseudopot_File cannot be empty")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate.resolve()
        raise ParsecInputError(
            f"Atom_Type {symbol}: Pseudopot_File {candidate} does not exist"
        )
    directories = [input_directory]
    if pseudopotential_directory is not None:
        directories.append(pseudopotential_directory)
    current = Path.cwd().resolve()
    if current not in directories:
        directories.append(current)
    tried = []
    for directory in directories:
        path = (directory / candidate).resolve()
        tried.append(path)
        if path.is_file():
            return path
    attempted = "\n  ".join(str(path) for path in tried)
    raise ParsecInputError(
        f"Atom_Type {symbol}: Pseudopot_File {raw!r} not found; searched:\n  {attempted}"
    )


@dataclass(frozen=True)
class _Input15:
    """What :func:`_input15_keywords` reads.

    ``hartree_controls`` replaces fields of :class:`HartreeSettings` (empty
    when the input gives neither Hartree CG keyword); ``complete_input`` is
    ``Generate_Complete_Parsec_In``.
    """

    warnings: tuple[str, ...]
    hartree_controls: dict
    complete_input: bool
    libxc: "_LibxcRequest | None" = None


@dataclass(frozen=True)
class _LibxcRequest:
    """A libxc functional asked for: ``LibXC_Functional`` (``items``) or 1.5's ``Func_Id_X``/``Func_Id_C``.

    ``func_ids`` are 1.5's integers (-1 when not given: 1.5's defaults, 0 =
    none); ``enforce_fhc`` is ``LibXC_Enforce_FHC`` (``None``: the default of
    :func:`_libxc_xc_functional`).
    """

    items: tuple[str, ...] | None
    func_ids: tuple[int, int] | None
    enforce_fhc: bool | None
    density_threshold: float | None = None
    tau_threshold: float | None = None


def _input15_keywords(scalar, one, optional_bool, *, states_text: str | None) -> _Input15:
    """Check the PARSEC 1.5 keywords of this layer (1.5's ``src/io/usrinputfile.F90``) with 1.5's validation.

    * ``Use_LibXC: .true.`` with ``Func_Id_X``/``Func_Id_C`` (3895-3948) and this
      port's ``LibXC_Functional`` (names or numbers) and ``LibXC_Enforce_FHC``
      are returned as a :class:`_LibxcRequest` (the functional is resolved
      against the library with the spin of the run, :func:`_libxc_xc_functional`);
      with ``Use_LibXC: .false.`` (1.5's default) ``Func_Id_X``/``Func_Id_C`` are
      read and ignored, as 1.5 does.
    * ``Hartree_CG_Tolerance``/``Hartree_CG_Criterion`` (2836-2842): written
      in the input, either one gives 1.5's Poisson CG, ``potential/hartree.F90``
      2096-2108: relative tolerance (1.5's default 1e-13), the test (2 against
      the right-hand side, any other value 1 against the initial residual),
      2400 matrix-vector products and the absolute tolerance 1e-13.  Without
      them this port keeps its CG (1e-7, 1600; kept as the default on
      2026-11-04).
    * The keywords of :data:`_INPUT15_INERT_KEYS` are read with their ESDF type
      and change nothing; one warning names them.  ``Test_Logic`` accepts
      ``none`` only (``forces``/``lanczos`` are 1.5's internal tests);
      ``Test_Logic`` and ``Rritz_Logic`` are compared case-sensitively, as 1.5
      reads them without ``esdf_reduce``.  The checks of 1.5's MPI layout
      (``Npcol`` against the number of processes, ``Nslices`` against
      ``Npcol``) have nothing to check here.
    * ``Output_Ascii_Data`` and ``Print_Unconverged_Density`` (``.true.``) are
      refused with the reason; ``Generate_Complete_Parsec_In`` is returned.
    """

    notes: list[str] = []
    # Use_LibXC and the libxc functional ids (usrinputfile.F90:3895-3948), and this
    # port's LibXC_Functional / LibXC_Enforce_FHC.
    func_ids = []
    id_values = [-1, -1]
    for index, (key, label) in enumerate((("func_id_x", "Func_Id_X"), ("func_id_c", "Func_Id_C"))):
        if key in scalar:
            id_values[index] = _integer(one(key), label=label)  # esdf_integer
            func_ids.append(label)
    use_libxc = optional_bool("use_libxc") if "use_libxc" in scalar else None
    enforce_fhc = optional_bool("libxc_enforce_fhc") if "libxc_enforce_fhc" in scalar else None
    density_threshold = None
    if "libxc_density_threshold" in scalar:
        density_threshold = _strict_float(one("libxc_density_threshold"), label="LibXC_Density_Threshold")
        if not (np.isfinite(density_threshold) and density_threshold > 0.0):
            raise ParsecInputError("LibXC_Density_Threshold must be positive (e/bohr^3)")
    tau_threshold = None
    if "libxc_tau_threshold" in scalar:
        tau_threshold = _strict_float(one("libxc_tau_threshold"), label="LibXC_Tau_Threshold")
        if not (np.isfinite(tau_threshold) and tau_threshold > 0.0):
            raise ParsecInputError("LibXC_Tau_Threshold must be positive (Hartree/bohr^3)")
    libxc = None
    if "libxc_functional" in scalar:
        if use_libxc is False:
            raise ParsecInputError(
                "LibXC_Functional names libxc functionals, Use_LibXC: .false. switches libxc off: "
                "give one of them"
            )
        if func_ids:
            raise ParsecInputError(
                "give the libxc functional as LibXC_Functional or as PARSEC 1.5's "
                f"{' and '.join(func_ids)}, not both"
            )
        items = tuple(item for item in re.split(r"[\s,+]+", one("libxc_functional").strip()) if item)
        if not items:
            raise ParsecInputError("LibXC_Functional names no functional")
        libxc = _LibxcRequest(
            items=items,
            func_ids=None,
            enforce_fhc=enforce_fhc,
            density_threshold=density_threshold,
            tau_threshold=tau_threshold,
        )
    elif use_libxc:
        libxc = _LibxcRequest(
            items=None,
            func_ids=(id_values[0], id_values[1]),
            enforce_fhc=enforce_fhc,
            density_threshold=density_threshold,
            tau_threshold=tau_threshold,
        )
    else:
        if enforce_fhc is not None or density_threshold is not None or tau_threshold is not None:
            raise ParsecInputError(
                "LibXC_Enforce_FHC, LibXC_Density_Threshold and LibXC_Tau_Threshold belong to a libxc "
                "functional (LibXC_Functional)"
            )
        if func_ids:
            notes.append(
                f"{' and '.join(func_ids)} name libxc functionals, which PARSEC 1.5 reads only with "
                "Use_LibXC: .true. (usrinputfile.F90:3909-3948); they are ignored, as there: the "
                "functional is that of Correlation_Type"
            )

    # Keywords that tune the parallel layout and libraries of PARSEC 1.5.
    given = [key for key in _INPUT15_INERT_KEYS if key in scalar]
    values: dict[str, object] = {}
    for key in given:
        label = _INPUT15_INERT_KEYS[key]
        kind = _INPUT15_INERT_TYPES[key]
        text = one(key)
        if kind == "logical":
            values[key] = _boolean(text, label=label)
        elif kind == "integer":
            values[key] = _integer(text, label=label)
        elif kind == "double":
            values[key] = _strict_float(text, label=label)
        else:
            values[key] = text.strip()
    if "npcol" in values and "mpi_groups" in scalar:
        raise ParsecInputError(
            "Npcol is PARSEC 1.5's name of MPI_Groups (the MPI column groups): both are given; "
            "provide one of them"
        )
    if "test_logic" in values:
        test = values["test_logic"]
        if test not in {"none", "forces", "lanczos"}:
            # usrinputfile.F90:2768-2784, ierr 130 (the string is not reduced)
            raise ParsecInputError(
                f"Test_Logic: unknown test_logic type {test!r}; PARSEC 1.5 reads none, forces or "
                "lanczos, in lower case (ierr 130)"
            )
        if test != "none":
            raise UnsupportedParsecOptionError(
                f"Test_Logic: {test} is an internal test of PARSEC 1.5 ('forces' leaves the SCF "
                "loop after the first occupations, parsec.F90:1501-1507; 'lanczos' runs nothing); "
                "it is not ported: give none"
            )
    if "rritz_logic" in values and values["rritz_logic"] not in _INPUT15_RRITZ_LOGIC:
        # usrinputfile.F90:2667-2735, ierr 128 (the string is not reduced)
        raise ParsecInputError(
            f"Rritz_Logic: unknown rritz logic type {values['rritz_logic']!r}; PARSEC 1.5 reads "
            + ", ".join(_INPUT15_RRITZ_LOGIC)
            + ", in lower case (ierr 128)"
        )
    plain = [key for key in given if key not in {"use_hdf5", "spectrum_slicing"}]
    if "spectrum_slicing" in given and not values["spectrum_slicing"]:
        plain.append("spectrum_slicing")
    if plain:
        notes.append(
            ", ".join(_INPUT15_INERT_KEYS[key] for key in plain)
            + " tune PARSEC 1.5's parallel layout and libraries (MPI column groups, ScaLAPACK, "
            "ELPA, space-filling curves, spectrum slicing, threaded start vectors, internal "
            "tests) and have no effect here"
        )
    groups = values.get("npcol", 1)
    if isinstance(groups, int) and groups > 1:
        block = values.get("scalapack_block_size", 8)
        try:
            states = _integer(states_text, label="States_Num") if states_text is not None else None
        except ParsecInputError:
            states = None
        if states is not None and isinstance(block, int) and block > 0 and states % (groups * block):
            rounded = (states + groups * block - 1) // (groups * block) * (groups * block)
            notes.append(
                f"Npcol={groups}: PARSEC 1.5 (with ScaLAPACK) rounds States_Num up to a multiple of "
                f"Npcol*ScaLAPACK_Block_Size, {rounded} instead of {states} "
                "(usrinputfile.F90:4875-4884); this run computes the States_Num of the input"
            )
    if values.get("spectrum_slicing"):
        notes.append(
            "Spectrum_Slicing: .true.: PARSEC 1.5's spectrum-slicing eigensolver (slices of the "
            "spectrum on separate MPI groups, ScaLAPACK) is not ported; this run uses the eigensolver "
            "of the input (Eigensolver), which converges to the same states"
        )
    if values.get("use_hdf5"):
        notes.append(
            "Use_HDF5: .true.: this port writes no HDF5 files (PARSEC 1.5's "
            "parsec_moveNNNNN_iterNNNNN.h5 and its restart.h5); it writes parsec.dat as with "
            "Use_HDF5: .false. (Output_Data), which Restart_Run reads.  A PARSEC 1.5 binary built "
            "without HDF5 writes neither file"
        )

    # The Poisson CG of potential/hartree.F90 (usrinputfile.F90:2836-2842).
    hartree_controls: dict = {}
    if "hartree_cg_tolerance" in scalar or "hartree_cg_criterion" in scalar:
        tolerance = _strict_float(
            one("hartree_cg_tolerance", "1.0d-13"), label="Hartree_CG_Tolerance"
        )
        if not tolerance > 0.0:
            raise ParsecInputError(
                f"Hartree_CG_Tolerance must be positive, got {tolerance:g}"
            )
        criterion = _integer(one("hartree_cg_criterion", "1"), label="Hartree_CG_Criterion")
        # usrinputfile.F90:2838: anything but 2 is the test against the initial residual
        criterion = 2 if criterion == 2 else 1
        hartree_controls = {
            "relative_tolerance": tolerance,
            "absolute_tolerance": 1.0e-13,
            "max_iterations": 2400,
            "cg_criterion": criterion,
        }
        notes.append(
            "Hartree_CG_Tolerance/Hartree_CG_Criterion: the Poisson CG of PARSEC 1.5 "
            f"(potential/hartree.F90): relative tolerance {tolerance:.1e}, test against the "
            + ("right-hand side" if criterion == 2 else "initial residual")
            + ", at most 2400 matrix-vector products (without these keywords this port stops at "
            "1e-7 of the initial residual and 1600 products)"
        )

    # The ASCII dumps of io/plaintext_io.F90 (usrinputfile.F90:2183-2188).
    for key, label, what in (
        (
            "output_ascii_data",
            "Output_Ascii_Data",
            "writes data.visit and one data_NNNNN.txt per MPI rank after the SCF (the grid points "
            "of the rank, rho, V_ion, V_H, V_xc and five orbitals each side of the gap)",
        ),
        (
            "print_unconverged_density",
            "Print_Unconverged_Density",
            "writes the same per-rank files at every SCF iteration",
        ),
    ):
        if optional_bool(key):
            raise UnsupportedParsecOptionError(
                f"{label}: .true. is not ported: PARSEC 1.5 {what}, in its fold-block layout "
                "(8 points per row, real wavefunctions of k-point 1 only); 1.5's own version "
                "segfaults with fewer than 5 occupied states and prints vxc(:,0), out of bounds. "
                "Output_Grid_Data writes the density, V_ion and V_H as cube files, and "
                "Save_Intermediate_Charge_Density the density of every iteration (parsec.dat)"
            )
    return _Input15(
        warnings=tuple(notes),
        hartree_controls=hartree_controls,
        complete_input=optional_bool("generate_complete_parsec_in"),
        libxc=libxc,
    )


def _libxc_xc_functional(
    request: _LibxcRequest, *, spin_polarized: bool, correlation_given: bool, notes: list[str]
) -> str:
    """The ``libxc:`` spec of a :class:`_LibxcRequest`, checked against the library.

    1.5's ids: ``-1`` takes 1.5's defaults (``Func_Id_X`` 1 = LDA_X, ``Func_Id_C``
    9 = LDA_C_PZ, 12 = LDA_C_PW with spin), ``0`` leaves the slot empty.  A
    missing library, a functional libxc does not have and the functionals this
    code refuses (:meth:`parsec_python.V_xc.libxc.LibxcFunctionalInfo.refusal`)
    stop the input; a meta-GGA needs the meta-GGA Hamiltonian
    (:mod:`parsec_python.V_xc.meta_gga`).
    """

    import importlib.util
    import warnings as python_warnings

    from ..V_xc.libxc import LibxcRefusal, canonical_libxc_functional, parse_libxc_spec
    from ..V_xc.libxc_library import LibxcUnavailable

    if request.items is not None:
        items: tuple = request.items
        source = "LibXC_Functional"
    else:
        exchange, correlation = request.func_ids
        default_correlation = correlation == -1
        exchange = 1 if exchange == -1 else exchange
        correlation = (12 if spin_polarized else 9) if correlation == -1 else correlation
        if default_correlation and exchange > 0:
            try:
                from ..V_xc.libxc import libxc_functional_info

                kind = libxc_functional_info(exchange).kind
            except (LibxcUnavailable, ValueError):
                kind = None
            if kind == "exchange-correlation":
                notes.append(
                    f"Func_Id_X: {exchange} is an exchange-correlation functional and Func_Id_C is not given: "
                    f"PARSEC 1.5's default correlation {correlation} ({'LDA_C_PW' if correlation == 12 else 'LDA_C_PZ'}) "
                    "is added to it, as 1.5 does; give Func_Id_C: 0 for the functional alone"
                )
        items = tuple(value for value in (exchange, correlation) if value != 0)
        source = "Use_LibXC (Func_Id_X/Func_Id_C)"
        if not items:
            raise ParsecInputError("Use_LibXC: .true. with Func_Id_X: 0 and Func_Id_C: 0 names no functional")
        notes.append(
            "Use_LibXC: PARSEC 1.5 evaluates the first SCF iteration of every libxc run with PBE "
            "(101 + 130, potential/hartree.F90:1932-1937); this port uses the libxc functional "
            "from the first iteration (a meta-GGA starts with PBE, as there) and adds the exact "
            "exchange of a libxc hybrid, which 1.5 leaves out"
        )
    try:
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always")
            spec = canonical_libxc_functional(
                items,
                enforce_fhc=request.enforce_fhc,
                density_threshold=request.density_threshold,
                tau_threshold=request.tau_threshold,
            )
        notes.extend(str(warning.message) for warning in caught)
    except LibxcUnavailable as error:
        raise UnsupportedParsecOptionError(f"{source}: {error}") from error
    except LibxcRefusal as error:
        raise UnsupportedParsecOptionError(f"{source}: {error}") from error
    except ValueError as error:
        raise ParsecInputError(f"{source}: {error}") from error
    parsed = parse_libxc_spec(spec)
    if parsed.is_meta_gga and importlib.util.find_spec("parsec_python.V_xc.meta_gga") is None:
        raise UnsupportedParsecOptionError(
            f"{source}: {spec} is a meta-GGA; it needs the kinetic-energy density of the orbitals "
            "and the meta-GGA Hamiltonian (parsec_python.V_xc.meta_gga), which this build does not have"
        )
    if request.enforce_fhc is not None and not parsed.is_meta_gga:
        notes.append("LibXC_Enforce_FHC acts on meta-GGAs only; it changes nothing for " + spec)
    if correlation_given:
        notes.append(
            f"Correlation_Type does not select the functional: it is libxc's {spec} ({source}), "
            "as in PARSEC 1.5 with Use_LibXC"
        )
    return spec


def _input15_pseudopotential_notes(
    specifications, atoms, states: int | None, xc_functional: str, potentials=None
) -> list[str]:
    """What PARSEC 1.5's ``io/pseudo.f90`` says of the pseudopotentials, as warnings.

    * A CA file in a PBE run: 1.5 warns ("correlation potential does not
      match", 482-492) and uses the file; so does this port
      (:func:`parsec_python.V_ion.load_pseudopotentials`).
    * The minimum number of states (886-896): 1.5 (and the tree before it)
      stops with ierr 213 when ``States_Num < N_val/2 + 2``, ``N_val`` the
      valence electrons of the neutral atoms (``elval1``, the sum of their
      ionic charges), and asks for at least ``int(N_val/2 + 2) + 1`` states.
      This port runs such an input (``states`` ``None``: not checked).

    Nothing is said where the pseudopotentials cannot be read here: the run
    reports that itself.
    """

    import warnings as python_warnings

    from ..V_ion import load_pseudopotentials, pseudopotential_matches_functional
    from ..V_ion.ionic_potential import (
        _CA_LABELS,
        hybrid_pseudopotential_family,
        hybrid_pseudopotential_labels,
        hybrid_pseudopotential_refusal,
    )

    hybrid_labels = hybrid_pseudopotential_labels(xc_functional)
    if potentials is None:
        try:
            with python_warnings.catch_warnings():
                python_warnings.simplefilter("ignore")
                # A hybrid's files are checked below, by name, before the run.
                potentials = load_pseudopotentials(
                    specifications,
                    xc_functional=xc_functional,
                    check_correlation=hybrid_labels is None,
                )
        except (OSError, ValueError, KeyError):
            return []
    notes = []
    if hybrid_labels is not None:
        # A hybrid reads the files of its semilocal family (PBE; B3PW91 also PW91).
        for symbol, potential in potentials.items():
            if pseudopotential_matches_functional(potential, xc_functional):
                continue
            label = potential.correlation.strip().lower()
            if label not in _CA_LABELS:
                raise UnsupportedParsecOptionError(
                    hybrid_pseudopotential_refusal(symbol, potential.source, label, xc_functional)
                )
            notes.append(
                f"Atom_Type {symbol}: {potential.source} was generated for {potential.correlation!r}, "
                f"not for {hybrid_pseudopotential_family(xc_functional)}, the pseudopotentials of the "
                f"hybrid {xc_functional}; it is used, as a PBE run uses a CA file"
            )
    if xc_functional == "pbe":
        for symbol, potential in potentials.items():
            if not pseudopotential_matches_functional(potential, "pbe"):
                notes.append(
                    f"Atom_Type {symbol}: {potential.source} was generated for "
                    f"{potential.correlation!r}, not for PBE; PARSEC 1.5 warns ('correlation "
                    "potential does not match', io/pseudo.f90) and uses the file, and so does this run"
                )
    if states is None:
        return notes
    try:
        electrons = float(sum(float(potentials[atom.symbol].ionic_charge) for atom in atoms))
    except (KeyError, AttributeError):
        return notes
    if float(states) < 0.5 * electrons + 2.0:
        notes.append(
            f"States_Num={states} is below PARSEC 1.5's minimum of N_val/2 + 2 = {0.5 * electrons + 2.0:g} "
            f"(N_val = {electrons:g} valence electrons of the neutral atoms): PARSEC stops here "
            f"(pseudo.f90 ierr 213, 'Increase number of states to at least {int(0.5 * electrons + 2.0) + 1}'), "
            "this run goes on with the States_Num of the input"
        )
    return notes


def _parse_parsec_input(
    path: str | Path,
    *,
    pseudopotential_directory: str | Path | None = None,
    routines: RoutineChoice | None = None,
    esdf_lenient: bool = False,
) -> ParsecInputTranslation:
    source = Path(path).expanduser().resolve()
    pp_directory = (
        Path(pseudopotential_directory).expanduser().resolve()
        if pseudopotential_directory is not None
        else None
    )
    items = _read_items(source)

    scalar: dict[str, list[_InputItem]] = {}
    blocks: dict[str, list[_InputItem]] = {}
    for item in items:
        target = blocks if item.is_block else scalar
        target.setdefault(item.key, []).append(item)
    # PARSEC 1.5's names of renamed keywords (_INPUT15_ALIASES).
    _apply_input15_aliases(scalar)
    # What PARSEC 1.5 writes into its bak_complete_parsec.in, read as 1.5 reads it (empty blocks, Kpoint_Unit,
    # format-dependent species labels of every species); notes added to the warnings below.
    bak15_notes = _input15_bak_constructs(scalar, blocks)

    species_keys = {
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
        "atomic_radius",
        # LDA+U of the d channel (usrinputfile.F90: esdf_physical, in Ry).
        "ldaplusu_u",
        "ldaplusu_j",
        "nonlinear_core_correction",
        "alpha_filter",
        "beta1_filter",
        "core_filter",
        # Radius of the Ylm-projected DOS (usrinputfile.F90 l.1913).
        "ylm_dos_cutoff",
    }
    block_keys = {
        "atom_coord",
        "domain_shape_parameters",
        "cell_shape",
        "monkhorst_pack_grid",
        "monkhorst_pack_shift",
        "initial_ncl_moment",
        "kpoint_list",
        "bandstruc",
        "electron_per_orbital",
        "point_coord",
        "save_wfn_bands",
        "super_cell",
        "super_cell_vac",
        # Charged sheets of a slab (usrinputfile.F90:2222-2299).
        "sheet_coord",
        # Non-self-consistent run and DOS (usrinputfile.F90 l.1056-1197).
        "nscf_kpoints",
        "nscf_kgrid",
        "nscf_kgrid_shift",
        "dos_mp_grid",
        "dos_mp_shift",
        # PARSEC.py: the lateral points of an AFM scan (_afm_scan_settings).
        *FDET_BLOCKS,
        # PARSEC 1.5: the grid shift of the input (usrinputfile.F90:1369-1382).
        "grid_shift",
    }
    accepted_global = {
        "restart_run",
        "restart_from_wfndat",
        "relax_restart",
        # Output files of usrinputfile.F90 (parsec.dat, eigen.dat, cube files).
        "output_data",
        "output_file_name",
        "output_eigenvalues",
        "save_intermediate_eigenvalues",
        "save_intermediate_charge_density",
        "output_grid_data",
        "output_gw",
        # Non-self-consistent run, DOS and the G-space Hartree solve
        # (parsec_python.SCF.nscf, parsec_python.Output.dos, Hartree.gspace).
        "nscf_kpoint_unit",
        "nscf_states_num",
        "nscf_fermi_level",
        "create_dos",
        "dos_pnum",
        "ylm_dos_l",
        # PARSEC.py extension: the projection of dos.F90 with its defects.
        "ylm_dos_fortran_projection",
        "calc_hartree_g",
        # PARSEC.py extensions: Fortran's G-space local potential of a 3-D
        # cell, its alpha term, and its potshift (parsec_python.V_ion.gspace_local).
        "periodic_local_potential",
        "periodic_alpha_term",
        "zero_average_vxc",
        "old_pseudopotential_format",
        "periodic_system",
        "boundary_conditions",
        "cluster_domain_shape",
        "boundary_sphere_radius",
        "grid_spacing",
        "coordinate_unit",
        "ignore_symmetry",
        # PARSEC 1.5: blocks of the domain (usrinputfile.F90:798-817).
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
        # Eigensolver controls of usrinputfile.F90:1310-1531.
        "chebyshev_filtering",
        "convergence_criterion_approach",
        "maximum_matvec",
        "fix_number_eigenvalues",
        "exp_comm",
        "max_lap_buffers",
        "mpi_groups",
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
        "mixing_preconditioner",
        "kerker_wavevector",
        "kerker_dielectric_constant",
        "solver_lpole",
        "full_hartree",
        "hartree_boundary_tolerance",
        "hartree_atomic_tail",
        "domain_energy_tolerance",
        "atom_types_num",
        "correlation_type",
        "electric_field",
        "polarizability",
        "add_point_charges",
        "point_typ_num",
        "pt_chg",
        "add_charged_sheets",
        "num_charged_sheets",
        "sheet_charge",
        "external_mag_field",
        "orb_mag",
        "complex_wfn",
        "symmetrize_charge_density",
        "output_all_states",
        "output_level",
        "old_interpolation_format",
        "double_grid_order",
        "minimization",
        "movement_num",
        "skip_force",
        # PARSEC 1.5 keywords of the forces, ionic loops and bands.
        *_SMALL15_ACCEPTED_GLOBAL,
        "total_atom_num",
        "ion_energy_diff",
        "force_min",
        "max_step",
        "min_step",
        "bfgs_number_corr",
        # PARSEC.py extension: a looser SCF criterion while the forces are large.
        "relaxation_scf_upscale",
        *_MOLECULAR_DYNAMICS_KEYS,
        "dynamic_diag_tol",
        # PARSEC.py extension: the start of the SCF loop after a move.
        "potential_extrapolation",
        # PARSEC.py extension: how often molecular dynamics writes its structures (see ParsecInputTranslation).
        "md_output_interval",
        # PARSEC.py extension: the spin channels of the accelerated driver in
        # the symmetry sectors (SCFSettings.spin_symmetry_sectors).
        "spin_symmetry_sectors",
        # PARSEC.py extension: the input structure relaxed with a
        # machine-learned force field first (parsec_python.MLFF).
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
        # PARSEC.py extension: a modular initial-density source.  These
        # labels do not alter the DFT grid or any post-initialization physics.
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
        # PARSEC.py extension: one post-SCF evaluation of a hybrid functional
        # on the converged orbitals (ARES ``Lrefine``).  It changes nothing
        # of the SCF.
        "energy_refinement",
        # PARSEC.py extension: the Fock loop of a hybrid Correlation_Type.
        *_HYBRID_KEYS,
        # Tkatchenko-Scheffler dispersion correction (usrinputfile.F90).
        "van_der_waals",
        "hf_one",
        "hf_two",
        "hf_three",
        # The routines of the run (parsec_python.Input.routines): Andrew
        # Grieder's by default, the Fortran-derived ones with .true.
        "use_fortran_routines",
        # Eigensolver, SCF-control and start-vector labels of PARSEC 1.5 and
        # the extensions of its round (parsec_python.Input.solver_options).
        *_SOLVER_1P5_KEYS,
        # FDET of PARSEC 1.5 and its PARSEC.py switches (_fdet_settings).
        *FDET_KEYWORDS,
    }
    # PARSEC 1.5 keywords of the input layer (_INPUT15_* above).
    species_keys |= _INPUT15_SPECIES_KEYS
    accepted_global |= _INPUT15_ACCEPTED_GLOBAL
    unknown_scalars = sorted(set(scalar).difference(species_keys | accepted_global))
    unknown_blocks = sorted(set(blocks).difference(block_keys))
    # Notes of the ESDF-lenient mode (--esdf-lenient), added to the warnings below.
    lenient_notes: list[str] = []
    if unknown_scalars or unknown_blocks:
        names = unknown_scalars + [f"begin {name}" for name in unknown_blocks]
        if not esdf_lenient:
            raise UnsupportedParsecOptionError(
                "unsupported or unknown PARSEC option(s): " + ", ".join(names)
            )
        # ESDF (both PARSEC trees) warns about a label it does not know and never prints the warning:
        # the label is ignored.  The lenient mode does the same and says so.
        for name in unknown_scalars:
            scalar.pop(name)
        for name in unknown_blocks:
            blocks.pop(name)
        lenient_notes.append(
            "--esdf-lenient: the label(s) "
            + ", ".join(names)
            + " are not keywords of this reader and are ignored, as PARSEC's ESDF ignores labels it does not know"
        )

    def one(key: str, default: str | None = None) -> str:
        occurrences = scalar.get(key, [])
        if len(occurrences) > 1 and key not in species_keys | {"pt_chg"}:
            lines = ", ".join(str(item.line) for item in occurrences)
            if not esdf_lenient:
                raise ParsecInputError(
                    f"duplicate {key} values on lines {lines}; provide it once"
                )
            # ESDF hands out the first occurrence of a label (its kw_index cursor).
            note = (
                f"--esdf-lenient: {key} is given on lines {lines}; the first value, "
                f"{occurrences[0].value!r}, is used, as PARSEC's ESDF does"
            )
            if note not in lenient_notes:
                lenient_notes.append(note)
        if occurrences:
            return str(occurrences[0].value)
        if default is None:
            raise ParsecInputError(f"required PARSEC option {key} is missing")
        return default

    def optional_bool(key: str, default: bool = False) -> bool:
        if key not in scalar:
            return default
        return _boolean(one(key), label=key)

    # PARSEC 1.5 keywords of the input layer: libxc, the Hartree CG, the
    # inert tuning keywords, the ASCII dumps, bak_complete_parsec.in.
    input15 = _input15_keywords(
        scalar,
        one,
        optional_bool,
        states_text=str(scalar["states_num"][0].value) if "states_num" in scalar else None,
    )

    # The routines of the run (parsec_python.Input.routines): decided once
    # from the labels here; a value that the default routines do not compute
    # raises UseFortranRoutines below (fall_back), and parse_parsec_input
    # starts again with the Fortran-derived routines.
    # The two sets of routines differ for periodic cells and for the forces and
    # ionic loops of spinor orbitals; every other input (a cluster) runs the same
    # code under both and keeps the default without a note.
    choice_matters = _routines_choice_matters(scalar, one, optional_bool)
    if routines is None:
        # Use_Fortran_Routines: .true., .false. or not given (None: the default of each feature)
        requested = optional_bool(_routines_module.KEYWORD) if _routines_module.KEYWORD in scalar else None
        routines = (
            decide_routines(scalar, blocks, requested=requested)
            if choice_matters or requested
            else _routines_module.DEFAULT
        )

    def fall_back(reason: str) -> None:
        """An input of the default routines that they do not compute: start again on the Fortran-derived ones."""
        if not routines.fortran and choice_matters:
            raise UseFortranRoutines(routines.with_reason(reason))

    def fortran_by_default(feature: str) -> None:
        """A feature whose default is the Fortran-derived routines (the parameter not given).

        Andrew Grieder's routines so far: start again on the Fortran-derived ones.  Already on them (a label his
        routines do not read): the choice records the feature, so that the run says it is the default.
        """
        nonlocal routines
        if not choice_matters or routines.requested is not None:
            return
        if not routines.fortran:
            raise UseFortranRoutines(routines.with_default(feature))
        routines = routines.with_default(feature)

    if not routines.fortran and choice_matters:
        refusal = _default_routines_value_refusal(scalar, one, optional_bool)
        if refusal is not None:
            fall_back(refusal)

    def input_relative_path(key: str, *, follow_links: bool = True) -> Path | None:
        """Resolve an optional quoted path relative to the input file.

        ``follow_links=False`` keeps symbolic links: an interpreter of a
        Linux venv is a link to the base interpreter, which run under its
        own name would not see the packages of the venv.
        """

        if key not in scalar:
            return None
        raw = one(key).strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
            raw = raw[1:-1]
        if not raw:
            raise ParsecInputError(f"{key} cannot be empty")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        return candidate.resolve() if follow_links else Path(os.path.abspath(candidate))

    # Restart_Run (usrinputfile.F90 l.361-368): the SCF loop starts from the
    # potential of parsec.dat (parsec_python.SCF.restart).
    restart_run = optional_bool("restart_run")
    # Restart_From_Wfndat (usrinputfile.F90 l.371-375): with Restart_Run the
    # restart reads wfn.dat (restart.F90) instead of parsec.dat; alone it is
    # read and changes nothing (parsec.F90 l.917-933).
    restart_from_wfndat = optional_bool("restart_from_wfndat")
    # Output_GW (usrinputfile.F90 l.2481-2545): the WFN, RHO and VXC files of
    # BerkeleyGW after the run (parsec_python.Output.berkeleygw).
    output_gw = optional_bool("output_gw")
    # Not a keyword of usrinputfile.F90, which ignores labels it does not
    # know: the format is chosen per species by Pseudopotential_Format.
    old_pseudopotential_format = optional_bool("old_pseudopotential_format")
    boundary = _normalize_label(one("boundary_conditions", "cluster"))
    # usrinputfile.F90:392-411: cluster|0d, wire|tube|1d, slab|2d, bulk|3d|pbc.
    periodicity = _BOUNDARY_PERIODICITY.get(boundary.replace(".", "").replace("-", ""))
    if periodicity is None:
        # usrinputfile.F90 error 101.
        raise ParsecInputError(
            f"Boundary_Conditions={boundary!r} is unknown; PARSEC accepts cluster "
            "(0d), wire (tube, 1d), slab (2d) and bulk (3d, pbc)"
        )
    is_periodic = periodicity > 0
    # Boundary_Conditions: wire or slab (Fortran pbc%per = 1 or 2).
    wire_or_slab = periodicity in (1, 2)
    # PARSEC (1.5 and the tree before it) never reads Periodic_System: the
    # periodicity is that of Boundary_Conditions.  Without the line it is
    # inferred from them (a bulk input without it, which this reader refused
    # before, now runs); a line that contradicts them is refused below.
    periodic_flag = optional_bool("periodic_system", is_periodic)
    if periodicity == 1:
        # the default routines have no wire
        fall_back("Boundary_Conditions: wire")
    if periodicity == 2:
        # a slab runs on the Fortran-derived routines unless the input says Use_Fortran_Routines: .false.
        fortran_by_default(_routines_module.SLAB_FEATURE)
    if periodicity == 2 and not optional_bool("periodic_system"):
        # Andrew Grieder's slab needs Periodic_System: .true. (PARSEC does not).  The routine decision keeps the
        # value of the line as before the inference (false without it), so that an existing slab input keeps its
        # routines: Use_Fortran_Routines: .false. without the line runs the Fortran-derived slab, as it did.
        fall_back("a slab without Periodic_System: .true.")
    # Andrew Grieder's slab (qian/slab_boundary): a 3-D periodic cell of PeriodicCell.dimensions = 2 for the code
    # below, which keeps the wire/slab path of the Fortran-derived routines for wire_or_slab.
    andrew_slab = periodicity == 2 and not routines.fortran
    if andrew_slab:
        wire_or_slab = False
    if not wire_or_slab and periodic_flag != is_periodic:
        raise UnsupportedParsecOptionError(
            "Periodic_System and Boundary_Conditions disagree: "
            "Periodic_System=true requires Boundary_Conditions=bulk or slab, and "
            "Boundary_Conditions=cluster requires Periodic_System=false"
        )
    spin_polarized = optional_bool("spin_polarization")
    noncollinear = optional_bool("non_collinear_magnetism")
    # Fortran only actually routes through the complex spinor path when
    # SO_from_scratch=true (or, a combination this port does not implement,
    # Non_Collinear_magnetism=true together with SCF_SO or a species'
    # SO_PSP).  SCF_SO=true alone is otherwise inert in Fortran -- it is only
    # consulted inside that NCL branch.  Treating SCF_SO=true here as also
    # requesting self-consistent SOC is a deliberate simplification (users
    # asking for "self-consistent spin-orbit" by either name get the same
    # behavior) rather than strict key-for-key Fortran fidelity.
    self_consistent_spin_orbit = optional_bool("so_from_scratch") or optional_bool(
        "scf_so"
    )
    # usrinputfile.F90:1490-1500: dynamic_diag_tol, switched off with
    # Restart_Run ("dynamic tolerance is not working with restart run!").
    dynamic_tolerance = optional_bool("dynamic_diag_tol")
    dynamic_tolerance_off_for_restart = bool(dynamic_tolerance and restart_run)
    if dynamic_tolerance_off_for_restart:
        dynamic_tolerance = False
    # usrinputfile.F90:2583: the local potential and its slope are
    # interpolated directly instead of as r*V and r**2*dV/dr; only the
    # cluster builders (ionpot.f90, forcloc.f90) read it.
    old_interpolation = optional_bool("old_interpolation_format")
    double_grid_order = _integer(one("double_grid_order", "1"), label="Double_Grid_Order")
    if double_grid_order < 1:
        # usrinputfile.F90 error 113.
        raise ParsecInputError(
            f"Double_Grid_Order must be greater than zero, got {double_grid_order}"
        )
    # usrinputfile.F90: Polarizability (false) and Electric_Field (0.001
    # ry/bohr/e, read whatever the flag says).  polar.f90 alone applies the
    # field: without Polarizability PARSEC ignores it.
    polarizability = optional_bool("polarizability")
    electric_field = _electric_field_value(
        one("electric_field", "0.001"), label="Electric_Field"
    )
    # A spin-polarized wire or slab with k-points: the seven field loops of
    # the spin k-point loop, each from the channel potentials of the loop
    # before (usrinputfile.F90 allows pbc%per <= 2; SCF/polarizability.py).
    if polarizability and is_periodic and not wire_or_slab:
        # usrinputfile.F90, error 156.
        raise UnsupportedParsecOptionError(
            "Cannot do polarizability with periodic boundary conditions"
        )

    # usrinputfile.F90:2308-2361: without the line a spin-polarized run takes
    # the Perdew-Wang LSDA ("pl"), an unpolarized one CA.
    correlation_text = one("correlation_type", "pl" if spin_polarized else "ca")
    correlation = _normalize_label(correlation_text)
    libxc_notes: list[str] = []
    if input15.libxc is not None:
        xc_functional = _libxc_xc_functional(
            input15.libxc,
            spin_polarized=spin_polarized or noncollinear,
            correlation_given="correlation_type" in scalar,
            notes=libxc_notes,
        )
    else:
        xc_functional = (
            _CORRELATION_LABELS.get(correlation)
            or _META_GGA_LABELS.get(correlation)
            or _hybrid_label(correlation_text)
        )
    if xc_functional is None:
        raise ParsecInputError(
            f"Correlation_Type={correlation!r} is unknown; PARSEC accepts "
            + ", ".join(sorted(_CORRELATION_LABELS))
            + "; the hybrid functionals are "
            + ", ".join(sorted(HYBRID_XC_FUNCTIONALS))
            + "; the meta-GGAs are "
            + ", ".join(sorted(_META_GGA_LABELS))
        )
    if xc_functional in UNPOLARIZED_ONLY_XC_FUNCTIONALS and (
        spin_polarized or noncollinear
    ):
        # exc_spn.f90 has no branch for these labels: Fortran would return
        # the spin densities as potentials and no energy.
        raise UnsupportedParsecOptionError(
            f"Correlation_Type={correlation!r} has no spin-polarized form "
            "(PARSEC's exc_spn.f90 does not implement it); it cannot be "
            "combined with Spin_Polarization or Non_Collinear_magnetism"
        )
    exact_exchange = _exact_exchange_settings(scalar, one, xc_functional)
    # Notes of the PARSEC 1.5 keywords (input layer), added to the warnings below.
    input15_notes = list(input15.warnings) + libxc_notes
    if correlation == "wb" and input15.libxc is None:
        # exc.f90:975-1045 (unpolarized) and 363-452 (spin), pbe.f90 new_exchpbe/new_corpbe.
        # The same discretisation as 1.5's wb for an unpolarized cluster without symmetry: H2O
        # -34.56585964 Ry here, -34.56585973 Ry with 1.5 wb, -34.56506807 Ry with 1.5 pb.
        # 1.5's spin wb is not the derivative of its energy (H atom
        # 2.5e-4 Ry too high) and its symmetry map is that of a scalar: not copied.
        input15_notes.append(
            "Correlation_Type: wb (PARSEC 1.5's PBE with the White-Bird variational potential) "
            "runs this port's PBE, whose potential is the exact derivative of the discrete PBE "
            "energy, as 1.5's wb for an unpolarized cluster without symmetry; with spin "
            "polarization or symmetry this port's potential stays the correct one (1.5's differs)"
        )
    # Read only for Casida-Salahub, as esdf_physical in the 'cs' case of
    # usrinputfile.F90 (default 0 Ry).  Fortran ignores the line silently
    # otherwise, and so does this reader once the value has been checked.
    ionization_energy_difference = 0.0
    if "ion_energy_diff" in scalar:
        value = _energy_rydberg(one("ion_energy_diff"), label="Ion_Energy_Diff")
        if xc_functional == "casida_salahub":
            ionization_energy_difference = value

    # Foldblock_Side_Length, Ignore_Symmetry and the Grid_Shift block (PARSEC 1.5).
    grid_notes: list[str] = []
    foldblock_side = _foldblock_side_length(
        scalar, one, periodicity=periodicity, grid_notes=grid_notes
    )
    ignore_symmetry = optional_bool("ignore_symmetry")
    if foldblock_side != 1 and not ignore_symmetry:
        # usrinputfile.F90:814-817: a fold block other than 1 switches the symmetry off.
        ignore_symmetry = True
        grid_notes.append(
            f"Foldblock_Side_Length {foldblock_side}: symmetry is switched off, as in PARSEC 1.5 "
            "(' WARNING! SYMMETRY IS AUTOMATICALLY SWITCHED OFF')"
        )
    requested_grid_shift = _grid_shift_block(blocks, grid_notes)

    periodic_cell = None
    general_periodic_grid = False
    monkhorst_pack_dimensions: tuple[int, int, int] | None = None
    monkhorst_pack_shift: tuple[float, float, float] | None = None
    # The Monkhorst_Pack_Grid and Monkhorst_Pack_Shift of a bulk cell on the default routines (a mesh of more than
    # one point), whose wedge _default_routines_wedge may take instead of the full list.
    default_routines_mesh: tuple | None = None
    kpoint_notes: list[str] = []
    explicit_kpoint_array: np.ndarray | None = None
    explicit_kpoint_weights: np.ndarray | None = None
    shape: str | None = None
    # Only the sphere of a cluster can leave its radius to the default rule.
    default_radius = False
    # Super_Cell / Super_Cell_Vac (None without the block) and PARSEC's
    # ``lattvec``: the cell of the input before them, the unit of
    # lattice-vector coordinates and of the replication.
    super_cell = _parse_super_cell_block(
        blocks,
        "super_cell",
        minimum=1,
        periodic=is_periodic,
        count=periodicity if wire_or_slab else 3,
    )
    super_cell_vacuum = _parse_super_cell_block(
        blocks,
        "super_cell_vac",
        minimum=0,
        periodic=is_periodic,
        count=periodicity if wire_or_slab else 3,
    )
    super_cell_notes: list[str] = []
    # Notes of the k-point input of a wire or slab.
    warnings_from_kpoints: list[str] = []
    if not is_periodic and ("super_cell" in blocks or "super_cell_vac" in blocks):
        super_cell_notes.append(
            "Super_Cell/Super_Cell_Vac are read but have no effect for a "
            "cluster, as in PARSEC."
        )
    primitive_vectors: np.ndarray | None = None
    if wire_or_slab:
        spacing = _physical_length(one("grid_spacing"), label="Grid_Spacing")
        periodic_cell = _parse_wire_slab_cell(scalar, blocks, periodicity)
        primitive_vectors = periodic_cell.lattice_vectors
        if super_cell is not None or super_cell_vacuum is not None:
            # usrinputfile.F90:723-773: the periodic vectors are replicated
            # and stretched (supercell(1:per), cell_vac(1:per)); the confined
            # unit vectors stay.  The atoms are copied below.
            replication = super_cell or (1, 1, 1)
            vacuum = super_cell_vacuum or (0, 0, 0)
            periodic_cell = PeriodicCell(
                lattice_vectors=supercell_lattice(
                    periodic_cell.lattice_vectors, replication, vacuum
                )
            )
            primitive_vectors = primitive_lattice(
                periodic_cell.lattice_vectors, replication, vacuum
            )
        if "boundary_sphere_radius" not in scalar:
            # usrinputfile.F90 error 109.
            raise ParsecInputError(
                "Boundary_Sphere_Radius is required for a wire or slab: the "
                "radius of the cylinder or the half height of the slab"
            )
        confinement_radius = _physical_length(
            one("boundary_sphere_radius"), label="Boundary_Sphere_Radius"
        )
        if not confinement_radius > 0.0:
            raise ParsecInputError("Boundary_Sphere_Radius must be positive")
        if double_grid_order > 1:
            raise UnsupportedParsecOptionError(
                "Double_Grid_Order > 1 is not implemented for a wire or slab"
            )
        grid = PeriodicGridSettings(
            spacing=spacing,
            expansion_order=_integer(
                one("expansion_order", "12"), label="Expansion_Order"
            ),
            # usrinputfile.F90:786-794: 0.5, zero along a periodic direction
            # not orthogonal to the other one, zero with Ignore_Symmetry, or
            # the Grid_Shift block of PARSEC 1.5.
            shift=periodic_grid_shift(
                periodic_cell.lattice_vectors,
                ignore_symmetry=ignore_symmetry,
                requested=requested_grid_shift,
                periodicity=periodicity,
            ),
            box_lengths=tuple(
                float(value)
                for value in np.linalg.norm(periodic_cell.lattice_vectors, axis=1)
            ),
            periodicity=periodicity,
            confinement_radius=confinement_radius,
            # wires and slabs of this path are those of the Fortran-derived routines
            fortran_routines=True,
        )
    if is_periodic:
        spacing = _physical_length(one("grid_spacing"), label="Grid_Spacing")
        if not wire_or_slab:
            # Andrew Grieder's slab: two in-plane vectors and the height 2R of its box (dimensions = 2)
            periodic_cell = _parse_cell_shape(scalar, blocks, slab=andrew_slab)
            primitive_vectors = periodic_cell.lattice_vectors
            if super_cell is not None or super_cell_vacuum is not None:
                replication = super_cell or (1, 1, 1)
                vacuum = super_cell_vacuum or (0, 0, 0)
                periodic_cell = PeriodicCell(
                    lattice_vectors=supercell_lattice(
                        periodic_cell.lattice_vectors, replication, vacuum
                    )
                )
                primitive_vectors = primitive_lattice(
                    periodic_cell.lattice_vectors, replication, vacuum
                )
            grid = PeriodicGridSettings(
                spacing=spacing,
                expansion_order=_integer(
                    one("expansion_order", "12"), label="Expansion_Order"
                ),
                box_lengths=tuple(np.diag(periodic_cell.lattice_vectors)),
                double_grid_order=double_grid_order,
                fortran_routines=routines.fortran,
            )
            if andrew_slab:
                # Andrew Grieder's slab: its own grid (Grid.pbc.build_slab_grid) reads no shift or points rule
                if requested_grid_shift is not None:
                    fall_back("Grid_Shift with a slab (Andrew Grieder's slab grid has no shift)")
                grid = replace(
                    grid,
                    box_lengths=tuple(
                        np.diag(periodic_cell.lattice_vectors)
                        if _lattice_is_diagonal(periodic_cell.lattice_vectors)
                        else np.linalg.norm(periodic_cell.lattice_vectors, axis=1)
                    ),
                )
            elif not routines.fortran:
                # Andrew Grieder's grid: the Fortran grid, half a step off the origin along vectors
                # orthogonal to the others (zero with Ignore_Symmetry), and ``adjust_step``'s even
                # number of points per direction; the lengths of the lattice vectors of an oblique cell.
                # The Grid_Shift block of PARSEC 1.5 replaces the shift (his grid reads any shift).
                grid = replace(
                    grid,
                    shift=(
                        fortran_grid_shift(periodic_cell.lattice_vectors, ignore_symmetry)
                        if requested_grid_shift is None
                        else periodic_grid_shift(
                            periodic_cell.lattice_vectors,
                            ignore_symmetry=ignore_symmetry,
                            requested=requested_grid_shift,
                        )
                    ),
                    points_rule="fortran",
                    box_lengths=tuple(
                        np.diag(periodic_cell.lattice_vectors)
                        if _lattice_is_diagonal(periodic_cell.lattice_vectors)
                        else np.linalg.norm(periodic_cell.lattice_vectors, axis=1)
                    ),
                )
            else:
                # The Fortran-derived routines: Fortran's grid of every cell (Grid.pbc.build_fortran_periodic_grid),
                # its shift (usrinputfile.F90:1363-1404: 0.5, zero with Ignore_Symmetry and along a lattice vector
                # not orthogonal to another one, or the Grid_Shift block) and the lengths of the lattice vectors.
                # Whether the cell takes the general grid (non-orthogonal lattice vectors, or steps that differ)
                # is known once the space group's fractional translations are (_periodic_symmetry below).
                grid = replace(
                    grid,
                    shift=periodic_grid_shift(
                        periodic_cell.lattice_vectors,
                        ignore_symmetry=ignore_symmetry,
                        requested=requested_grid_shift,
                    ),
                )
                general_periodic_grid = uses_general_periodic_grid(periodic_cell, grid)
                grid = _fortran_bulk_box_lengths(grid, periodic_cell, general_periodic_grid)
        kpoint_method = _normalize_label(one("kpoint_method", "none"))
        if kpoint_method == "mp":
            mp_blocks = blocks.get("monkhorst_pack_grid", [])
            if len(mp_blocks) != 1 or len(mp_blocks[0].value) != 1:
                raise ParsecInputError(
                    "Kpoint_Method=mp requires exactly one Monkhorst_Pack_Grid "
                    "block with exactly one line of three integers"
                )
            values, unit = _block_numbers(mp_blocks[0].value, label="Monkhorst_Pack_Grid")
            if wire_or_slab and not unit and len(values) >= periodicity:
                # usrinputfile.F90:975-990 reads the first per entries; the
                # confined directions have one point.
                if any(value != 1 for value in values[periodicity:3]):
                    warnings_from_kpoints.append(
                        "Monkhorst_Pack_Grid: the entries beyond the periodic "
                        "direction(s) of a wire or slab are ignored, as in PARSEC"
                    )
                values = list(values[:periodicity]) + [1.0] * (3 - periodicity)
            if andrew_slab and len(values) == 2:
                values = [*values, 1.0]  # a slab samples its two periodic directions
            if len(values) != 3 or unit or any(value <= 0 for value in values):
                raise ParsecInputError(
                    "Monkhorst_Pack_Grid must contain exactly "
                    + ("two (slab) or three" if andrew_slab else "three")
                    + " positive integers"
                )
            if andrew_slab and values[2] != 1.0:
                # Andrew Grieder's slab has one k-point along z (PARSEC ignores the third number)
                fall_back("a slab Monkhorst_Pack_Grid with more than one point along z")
            monkhorst_pack_dimensions = tuple(int(value) for value in values)
            if any(value != original for value, original in zip(monkhorst_pack_dimensions, values)):
                raise ParsecInputError("Monkhorst_Pack_Grid values must be integers")
            if "monkhorst_pack_shift" in blocks:
                if periodicity == 3:
                    # a bulk cell's shifted grid runs on the Fortran-derived routines (Fortran's wedge and weights)
                    # unless the input says Use_Fortran_Routines: .false.
                    fortran_by_default(_routines_module.SHIFTED_MESH_FEATURE)
                if not routines.fortran:
                    # Andrew Grieder's routines (qian/slab_boundary, bulk cells and slabs; Use_Fortran_Routines:
                    # .false. for both):
                    # Fortran (irrbz.f90): the points are k = (i + shift)/n, i = 0 ... n-1, in coordinates of the
                    # reciprocal lattice vectors, so a zero shift gives a Gamma-centred mesh.  The mesh is kept as an
                    # explicit list of equally weighted points, except where a bulk cell's mesh is reduced to its
                    # irreducible wedge below (_default_routines_wedge).
                    shift_blocks = blocks["monkhorst_pack_shift"]
                    if len(shift_blocks) != 1 or len(shift_blocks[0].value) != 1:
                        raise ParsecInputError("Monkhorst_Pack_Shift needs one line of numbers")
                    shift_values, shift_unit = _block_numbers(shift_blocks[0].value, label="Monkhorst_Pack_Shift")
                    if andrew_slab and len(shift_values) == 2:
                        shift_values = [*shift_values, 0.0]
                    if len(shift_values) != 3 or shift_unit:
                        raise ParsecInputError("Monkhorst_Pack_Shift must contain " + ("two (slab) or three" if andrew_slab else "three") + " numbers")
                    axes = [
                        (np.arange(count) + offset) / count
                        for count, offset in zip(monkhorst_pack_dimensions, shift_values)
                    ]
                    mesh = np.meshgrid(*axes, indexing="ij")
                    # Folded into (-1/2, 1/2] (Grid.kpoints.fold_to_centered_zone, PARSEC 1.5's intent; the Bloch
                    # operator of the finite-difference grid is not invariant under k -> k + G).
                    from ..Grid.kpoints import fold_to_centered_zone

                    fractions = fold_to_centered_zone(np.column_stack([m.reshape(-1) for m in mesh]))
                    reciprocal = 2.0 * np.pi * np.linalg.inv(periodic_cell.lattice_vectors).T
                    explicit_kpoint_array = fractions @ reciprocal
                    explicit_kpoint_weights = np.full(len(fractions), 1.0 / len(fractions))
                    if periodicity == 3 and len(fractions) > 1:
                        # A bulk cell's mesh may be reduced to the wedge of the operations that map this grid onto
                        # itself, once the rest of the input is known (_default_routines_wedge).
                        default_routines_mesh = (
                            monkhorst_pack_dimensions,
                            tuple(float(value) for value in shift_values),
                        )
                    monkhorst_pack_dimensions = None
                else:
                    # usrinputfile.F90:977-993: one line, the shift of each
                    # periodic direction, >= 0 (errors 116, 117).  With it the
                    # grid is Fortran's ((i-1)+s)/n, reduced to the irreducible
                    # wedge (irrbz.f90); without it this port's own grid is kept.
                    shift_blocks = blocks["monkhorst_pack_shift"]
                    if len(shift_blocks) != 1 or len(shift_blocks[0].value) != 1:
                        raise ParsecInputError(
                            "error in specifying Monkhorst-Pack Shift: give one "
                            "Monkhorst_Pack_Shift block with one line (PARSEC error 116)"
                        )
                    values, unit = _block_numbers(
                        shift_blocks[0].value, label="Monkhorst_Pack_Shift"
                    )
                    if wire_or_slab and not unit and len(values) >= periodicity:
                        # usrinputfile.F90:986-987 reads mpshift(1:per); the
                        # confined directions keep zero.
                        if any(value != 0.0 for value in values[periodicity:3]):
                            warnings_from_kpoints.append(
                                "Monkhorst_Pack_Shift: the entries beyond the periodic "
                                "direction(s) of a wire or slab are ignored, as in PARSEC"
                            )
                        values = list(values[:periodicity]) + [0.0] * (3 - periodicity)
                    if unit or len(values) != 3 or any(value < 0.0 for value in values):
                        raise ParsecInputError(
                            "error in specifying Monkhorst-Pack Shift: three "
                            "non-negative numbers are required (PARSEC error 117)"
                        )
                    monkhorst_pack_shift = tuple(float(value) for value in values)
        elif kpoint_method == "manual":
            list_blocks = blocks.get("kpoint_list", [])
            file_path = input_relative_path("kpoint_file")
            if len(list_blocks) > 1:
                raise ParsecInputError("provide a single Kpoint_List block")
            if bool(list_blocks) == (file_path is not None):
                raise ParsecInputError(
                    "Kpoint_Method=manual requires exactly one of a Kpoint_List "
                    "block or a Kpoint_File (rows of 'kx ky kz [weight]')"
                )
            try:
                if list_blocks:
                    rows = []
                    for line in list_blocks[0].value:
                        values, unit = _block_numbers((line,), label="Kpoint_List")
                        if unit or len(values) not in (3, 4):
                            raise ParsecInputError(
                                f"line {list_blocks[0].line}: every Kpoint_List row "
                                "must be 'kx ky kz' or 'kx ky kz weight'"
                            )
                        rows.append(list(values))
                    if len({len(row) for row in rows}) != 1:
                        raise ParsecInputError(
                            "every Kpoint_List row must have the same number of columns"
                        )
                    raw_kpoints = np.asarray(rows, dtype=float)
                else:
                    raw_kpoints = read_kpoint_file(file_path)
                explicit_kpoint_array, explicit_kpoint_weights = explicit_kpoints(
                    raw_kpoints,
                    unit=_normalize_label(one("kpoint_unit", "cartesian_inverse_bohr")),
                    # Fortran converts with the reciprocal vectors of the
                    # input cell (``lattvec``), not of a Super_Cell.
                    cell=(
                        periodic_cell
                        if super_cell is None and super_cell_vacuum is None
                        else PeriodicCell(lattice_vectors=primitive_vectors)
                    ),
                )
            except (OSError, ValueError) as error:
                raise ParsecInputError(f"Kpoint_Method=manual: {error}") from error
            if wire_or_slab and np.any(
                np.abs(explicit_kpoint_array[:, periodicity:]) > 1.0e-12
            ):
                raise ParsecInputError(
                    "Kpoint_List: a wire or slab has Bloch vectors along its periodic "
                    "direction(s) only; the k-point components along the confined "
                    "direction(s) must be zero"
                )
            # Unequal weights are normalized to one (init_var.F90:365-367);
            # the SCF drivers weight every level by its k-point (flevel.f90).
            if periodicity == 3 and not np.allclose(
                explicit_kpoint_weights, explicit_kpoint_weights[0], rtol=1.0e-12, atol=0.0
            ):
                # a bulk cell's weighted list runs on the Fortran-derived routines (their grid and space group, as
                # before 2026-11-08) unless the input says Use_Fortran_Routines: .false.
                fortran_by_default(_routines_module.WEIGHTED_LIST_FEATURE)
        elif kpoint_method != "none":
            raise ParsecInputError(
                f"Kpoint_Method must be 'none', 'mp', or 'manual', got {kpoint_method!r}"
            )
        if kpoint_method != "manual" and (
            "kpoint_list" in blocks or "kpoint_file" in scalar or "kpoint_unit" in scalar
        ):
            raise ParsecInputError(
                "Kpoint_List/Kpoint_File/Kpoint_Unit require Kpoint_Method: manual"
            )
        if kpoint_method != "mp" and "monkhorst_pack_shift" in blocks:
            kpoint_notes.append(
                "Monkhorst_Pack_Shift is read only with Kpoint_Method: mp; it is "
                "ignored, as in PARSEC"
            )
    else:
        if "monkhorst_pack_shift" in blocks:
            kpoint_notes.append(
                "Monkhorst_Pack_Shift applies to periodic systems; it is ignored "
                "for a cluster, as in PARSEC"
            )
        if "kpoint_list" in blocks or "kpoint_file" in scalar or "kpoint_unit" in scalar:
            raise ParsecInputError(
                "Kpoint_List/Kpoint_File/Kpoint_Unit apply only to periodic systems"
            )
        shape = _normalize_label(one("cluster_domain_shape", "sphere"))
        spacing = _physical_length(one("grid_spacing"), label="Grid_Spacing")
        # A sphere whose radius the input leaves out, or gives as "auto", gets
        # it from the default rule, below, once the atoms and the charge are
        # known.
        default_radius = (
            shape == "sphere"
            and _normalize_label(one("boundary_sphere_radius", "auto")) == "auto"
        )
        if default_radius:
            radius = None
            box_lengths = None
        elif shape == "sphere":
            radius = _physical_length(
                one("boundary_sphere_radius"), label="Boundary_Sphere_Radius"
            )
            box_lengths = None
        elif shape == "box":
            domain_blocks = blocks.get("domain_shape_parameters", [])
            if len(domain_blocks) != 1:
                raise ParsecInputError(
                    "box domain requires exactly one Domain_Shape_Parameters block"
                )
            values, unit = _block_numbers(
                domain_blocks[0].value, label="Domain_Shape_Parameters"
            )
            if len(values) != 3:
                raise ParsecInputError(
                    "box Domain_Shape_Parameters must contain three full side lengths"
                )
            # PARSEC reads this block directly in bohr. Permit an explicit unit as
            # a convenience, while retaining raw-bohr behavior when absent.
            factor = (
                _physical_length(f"1 {unit}", label="Domain_Shape_Parameters")
                if unit
                else 1.0
            )
            box_lengths = tuple(float(value * factor) for value in values)
            radius = 0.5 * float(np.linalg.norm(box_lengths))
        elif shape in {"ellipsoid", "cylinder"}:
            # usrinputfile.F90 l.559-653: one line of Domain_Shape_Parameters,
            # x/y/z radii or radius, length and orientation (1, 2, 3); error 110.
            domain_blocks = blocks.get("domain_shape_parameters", [])
            if len(domain_blocks) != 1 or len(domain_blocks[0].value) != 1:
                raise ParsecInputError(
                    f"A Domain_Shape_Parameters block of one line is required for "
                    f"{shape} cluster domains (PARSEC error 110)"
                )
            values, unit = _block_numbers(
                domain_blocks[0].value, label="Domain_Shape_Parameters"
            )
            if len(values) < 3:
                raise ParsecInputError(
                    "bad specification of the "
                    + (
                        "ellipsoidal radii: x_radius y_radius z_radius"
                        if shape == "ellipsoid"
                        else "cylindrical boundaries: radius length orientation"
                    )
                    + " (PARSEC error 110)"
                )
            factor = (
                _physical_length(f"1 {unit}", label="Domain_Shape_Parameters")
                if unit
                else 1.0
            )
            if shape == "ellipsoid":
                parameters = tuple(float(value * factor) for value in values[:3])
                if min(parameters) <= 0.0:
                    raise ParsecInputError(
                        "bad specification of ellipsoidal radii (PARSEC error 110)"
                    )
                radius = max(parameters)
            else:
                axis = values[2]
                if values[0] <= 0.0 or values[1] <= 0.0:
                    raise ParsecInputError(
                        "bad specification of the cylindrical radius or length "
                        "(should be positive; PARSEC error 110)"
                    )
                if axis not in (1, 2, 3):
                    raise ParsecInputError(
                        "bad specification of cylindrical orientation: 1 for the x "
                        "axis, 2 for y, 3 for z (PARSEC error 110)"
                    )
                parameters = (float(values[0] * factor), float(values[1] * factor), float(axis))
                radius = float(np.sqrt((parameters[1] / 2.0) ** 2 + parameters[0] ** 2))
            box_lengths = None
            domain_parameters = parameters
        else:
            raise UnsupportedParsecOptionError(
                f"Cluster_Domain_Shape={shape!r}; PARSEC's shapes are sphere, "
                "ellipsoid, cylinder and box"
            )

        # usrinputfile.F90:1363-1382 (PARSEC 1.5): 0.5, zero with Ignore_Symmetry, or the Grid_Shift block (its
        # absolute values; a cluster has no periodic direction to zero).
        if requested_grid_shift is not None:
            shift = tuple(float(value) for value in requested_grid_shift)
        else:
            shift = (0.0, 0.0, 0.0) if ignore_symmetry else (0.5, 0.5, 0.5)
        grid = GridSettings(
            spacing=spacing,
            # The settings are checked here whatever the radius: with the
            # default rule a stand-in, which the rule replaces below.
            radius=spacing if default_radius else radius,
            expansion_order=_integer(
                one("expansion_order", "12"), label="Expansion_Order"
            ),
            shift=shift,
            domain_shape=shape,
            box_lengths=box_lengths,
            double_grid_order=double_grid_order,
            **(
                {"domain_parameters": domain_parameters}
                if shape in {"ellipsoid", "cylinder"}
                else {}
            ),
            # Foldblock_Side_Length (PARSEC 1.5): the domain filled up to whole blocks.
            foldblock_side_length=foldblock_side,
        )

    # Coordinate_Unit (usrinputfile.F90:1685-1720): lattice_vectors gives
    # r = sum_j f_j a_j with the vectors of the input cell (``lattvec``).
    coordinate_matrix: np.ndarray | None = None
    coordinate_factor = 1.0
    fractional_coordinates = False
    if _normalize_label(one("coordinate_unit", "cartesian_bohr")) in {
        "lattice_vectors",
        "latticevectors",
    }:
        if not is_periodic:
            raise ParsecInputError(
                "Coordinate_Unit: lattice_vectors needs a periodic "
                "Boundary_Conditions; a cluster has no lattice vectors"
            )
        coordinate_matrix = primitive_vectors
        # Andrew Grieder's routines fold fractional coordinates into the cell (below).
        fractional_coordinates = not routines.fortran
    else:
        coordinate_factor = _coordinate_factor(
            one("coordinate_unit", "cartesian_bohr")
        )
    # Super_Cell: every input atom is followed by its copies.
    replicate = super_cell is not None and int(np.prod(super_cell)) != 1
    symbols = [str(item.value).strip() for item in scalar.get("atom_type", [])]
    declared_types = _integer(one("atom_types_num"), label="Atom_Types_Num")
    if declared_types != len(symbols):
        raise ParsecInputError(
            f"Atom_Types_Num={declared_types}, but {len(symbols)} Atom_Type "
            "occurrences were found"
        )
    if len(set(symbols)) != len(symbols):
        raise ParsecInputError(
            "each Atom_Type must have a unique element symbol"
            + (
                " (two species of one element: give them labels such as C_a and C_b, each with "
                "its Pseudopot_File)"
                if "pseudopot_file" in scalar
                else ""
            )
        )
    for key in species_keys.difference({"atom_type"}):
        occurrences = scalar.get(key, [])
        if len(occurrences) > declared_types:
            lines = ", ".join(str(item.line) for item in occurrences)
            raise ParsecInputError(
                f"{key} occurs {len(occurrences)} times on lines {lines}, "
                f"but Atom_Types_Num is {declared_types}"
            )
    coordinate_blocks = blocks.get("atom_coord", [])
    if len(coordinate_blocks) != declared_types:
        raise ParsecInputError(
            f"expected {declared_types} Atom_Coord blocks, found {len(coordinate_blocks)}"
        )

    def species_value(key: str, index: int, default: str | None = None) -> str:
        occurrences = scalar.get(key, [])
        if index < len(occurrences):
            return str(occurrences[index].value)
        if default is None:
            raise ParsecInputError(
                f"Atom_Type {symbols[index]} is missing occurrence {index + 1} of {key}"
            )
        return default

    type_lines = [item.line for item in scalar.get("atom_type", [])]

    def require_own_section(key: str, label: str, none_hint: str = " (0 for none)") -> None:
        """Refuse a species keyword ESDF would hand to another species.

        ``usrinputfile.F90`` reads it once per Atom_Type and ESDF returns the
        next occurrence each time, so the n-th line goes to the n-th
        Atom_Type wherever it stands.
        """

        for position, item in enumerate(scalar.get(key, [])):
            section = sum(1 for line in type_lines if line < item.line) - 1
            section = max(section, 0)
            if section != position:
                raise ParsecInputError(
                    f"line {item.line}: {label} stands after Atom_Type "
                    f"{symbols[section]}, but PARSEC gives the n-th {label} of "
                    "the input to the n-th Atom_Type, which is "
                    f"{symbols[position]} for this one; give every Atom_Type "
                    f"before it its own {label}{none_hint}"
                )

    require_own_section("ldaplusu_u", "ldaplusu_u")
    require_own_section("ldaplusu_j", "ldaplusu_j")
    # PARSEC 1.5 (usrinputfile.F90:3164): read for every Atom_Type, the n-th to the n-th
    require_own_section("pseudopot_file", "Pseudopot_File", " (its default file name for the default)")

    moment_blocks = blocks.get("initial_ncl_moment", [])
    if moment_blocks and not noncollinear:
        raise ParsecInputError(
            "Initial_NCL_Moment blocks require Non_Collinear_magnetism: .true."
        )
    if moment_blocks and len(moment_blocks) != declared_types:
        raise ParsecInputError(
            f"expected {declared_types} Initial_NCL_Moment blocks (one per "
            f"Atom_Type, in order), found {len(moment_blocks)}"
        )
    minimization = _normalize_label(one("minimization", "none"))
    if minimization == "":
        minimization = "none"
    if minimization not in {"none", "simple", "bfgs", "manual"}:
        raise ParsecInputError(f"unknown minimization method: {minimization!r}")
    # usrinputfile.F90 reads the molecular dynamics input with the
    # minimization: the two exclude each other, and either one makes the
    # movable flags of the atoms count.
    molecular_dynamics = MolecularDynamicsSettings.from_keywords(
        {
            key: one(key)
            for key in (*_MOLECULAR_DYNAMICS_KEYS, "minimization", "polarizability")
            if key in scalar
        }
    )
    atoms_move = minimization != "none" or molecular_dynamics.enabled

    local_map = {"s": 0, "p": 1, "d": 2, "f": 3}
    atoms: list[Atom] = []
    movable: list[bool] = []
    coordinate_notes: list[str] = []
    specifications: dict[str, SpeciesPotential] = {}
    global_cubic_occurrences = scalar.get("cubic_spline", [])
    old_format_species = 0
    hubbard_notes: list[str] = []
    # The position of each format-dependent label's cursor (FORMATS_READING).
    cursors: dict[str, int] = {}
    formats: list[str] = []
    filter_values: list[float] = []
    # PARSEC 1.5 reads Potential_Num for martins_old and fhipp only: a line too many would shift the cursor, so the
    # input of the PARSEC before 1.5 (a line for every species) is refused before a species takes a wrong one.
    given_formats = [
        _normalize_label(species_value("pseudopotential_format", index, "martins_new"))
        for index in range(len(symbols))
    ]
    readers_of_potential_num = sum(name in FORMATS_READING["potential_num"] for name in given_formats)
    known_formats = all(name in POTENTIAL_FILE_SUFFIX or name == "martinsnew" for name in given_formats)
    # (an unknown format is error 134 of the species loop, which comes first in usrinputfile.F90)
    if known_formats and len(scalar.get("potential_num", [])) > readers_of_potential_num:
        raise ParsecInputError(
            f"Potential_Num occurs {len(scalar['potential_num'])} times, but only {readers_of_potential_num} "
            "Atom_Type(s) have Pseudopotential_Format martins_old or fhipp, the formats "
            "PARSEC 1.5 reads it for (usrinputfile.F90:3244-3248), giving the n-th "
            "Potential_Num to the n-th of them; remove the lines of the other species "
            "(the PARSEC before 1.5 read one for every species)"
        )
    for species_index, (symbol, coordinate_item) in enumerate(
        zip(symbols, coordinate_blocks)
    ):
        pseudo_format = _normalize_label(
            species_value("pseudopotential_format", species_index, "martins_new")
        )
        if pseudo_format == "martinsnew":
            pseudo_format = "martins_new"
        if pseudo_format not in POTENTIAL_FILE_SUFFIX:
            # usrinputfile.F90 error 134.
            raise ParsecInputError(
                f"Atom_Type {symbol}: unknown Pseudopotential_Format "
                f"{pseudo_format!r}; PARSEC reads martins_new, martins_old, "
                "martins_wang and fhipp"
            )
        formats.append(pseudo_format)
        local_label = None
        if pseudo_format in FORMATS_READING["local_component"]:
            # No Local_Component for a Martins-Wang species: its file names
            # the local channel (usrinputfile.F90:1806-1827).
            local_index = cursors.get("local_component", 0)
            cursors["local_component"] = local_index + 1
            local_occurrences = scalar.get("local_component", [])
            if local_index >= len(local_occurrences):
                raise ParsecInputError(
                    f"Atom_Type {symbol} is missing occurrence {local_index + 1} "
                    "of local_component"
                )
            local_label = _normalize_label(str(local_occurrences[local_index].value))
            if local_label not in local_map:
                raise ParsecInputError(
                    f"Atom_Type {symbol}: Local_Component must be s, p, d, or f"
                )
        spin_orbit = _boolean(
            species_value("so_psp", species_index, "false"),
            label=f"{symbol} SO_PSP",
        )
        read_vcd = _boolean(
            species_value("read_vcd", species_index, "false"),
            label=f"{symbol} Read_VCD",
        )
        use_spline = (
            _boolean(
                str(global_cubic_occurrences[species_index].value),
                label=f"{symbol} Cubic_Spline",
            )
            if species_index < len(global_cubic_occurrences)
            else False
        )
        format_input: dict[str, object] = {}
        if pseudo_format != "martins_new":
            if read_vcd:
                # usrinputfile.F90 error 234.
                raise ParsecInputError(
                    f"Atom_Type {symbol}: Read_VCD needs a martins_new "
                    "pseudopotential; an old Martins file holds no valence "
                    "charge density"
                    if pseudo_format == "martins_old"
                    else f"Atom_Type {symbol}: Read_VCD needs a martins_new "
                    f"pseudopotential; a {pseudo_format} file holds no valence "
                    "charge density"
                )
            if spin_orbit and pseudo_format in {"martins_wang", "fhipp"}:
                # usrinputfile.F90 error 135 for FHIPP; pseudo.f90 reads the
                # spin-orbit columns of a Martins-Wang file into the wrong
                # array index.
                raise UnsupportedParsecOptionError(
                    f"Atom_Type {symbol}: SO_PSP is not supported for a "
                    f"{pseudo_format} pseudopotential"
                )
            format_input = _format_input(
                symbol, pseudo_format, scalar, blocks, species_index, cursors
            )
        if species_index < len(scalar.get("pseudopot_file", [])):
            # PARSEC 1.5's Pseudopot_File of this species (usrinputfile.F90:3164-3188)
            potential_path = _species_potential_file(
                str(scalar["pseudopot_file"][species_index].value),
                symbol,
                source.parent,
                pp_directory,
            )
        else:
            potential_path = _find_potential(
                symbol,
                source.parent,
                pp_directory,
                POTENTIAL_FILE_SUFFIX[pseudo_format],
            )
        if pseudo_format == "martins_wang":
            try:
                local_l = read_martins_wang_header(potential_path).local_angular_momentum
            except (OSError, ValueError) as error:
                raise ParsecInputError(f"Atom_Type {symbol}: {error}") from error
        else:
            local_l = local_map[local_label]
        if is_periodic:
            # usrinputfile.F90:1897-1907 reads the filters of every species of
            # a periodic system; Core_Filter defaults to Alpha_Filter.
            alpha = _strict_float(
                species_value("alpha_filter", species_index, "0"),
                label=f"{symbol} Alpha_Filter",
            )
            beta1 = _strict_float(
                species_value("beta1_filter", species_index, "0"),
                label=f"{symbol} Beta1_Filter",
            )
            core_text = species_value("core_filter", species_index, "")
            core = (
                _strict_float(core_text, label=f"{symbol} Core_Filter")
                if core_text
                else None
            )
            if core == 0.0 and alpha == 0.0 and beta1 == 0.0:
                # Core_Filter 0 without the other two (PARSEC 1.5's bak_complete_parsec.in writes it so):
                # no Fourier filter, as without the label (routines.NO_EFFECT).
                core = None
            for name, value in (
                ("Alpha_Filter", alpha),
                ("Beta1_Filter", beta1),
                ("Core_Filter", 0.0 if core is None else core),
            ):
                if value < 0.0:
                    raise ParsecInputError(
                        f"Atom_Type {symbol}: {name} must be zero or positive"
                    )
            if alpha != 0.0 or beta1 != 0.0 or core is not None:
                format_input.update(
                    fourier_filter_alpha=alpha,
                    fourier_filter_beta1=beta1,
                    fourier_filter_core=core,
                )
        else:
            for key in ("alpha_filter", "beta1_filter", "core_filter"):
                text_value = species_value(key, species_index, "")
                if text_value:
                    filter_values.append(abs(_first_float(text_value, label=key)))
        element_symbol = species_value(
            "element_symbol", species_index, ""
        ).strip()
        if not element_symbol:
            match = re.match(r"^([A-Z][a-z]?)(?:[-_].*)?$", symbol)
            if match is None:
                raise ParsecInputError(
                    f"Atom_Type {symbol!r} is not a recognizable element/species label; "
                    "provide Element_Symbol for this species"
                )
            element_symbol = match.group(1)
        correction = _energy_rydberg(
            species_value("atomic_energy_correction", species_index, "0 Ry"),
            label=f"{symbol} Atomic_Energy_Correction",
        )
        # usrinputfile.F90 reads it for every species, whatever the spin
        # setting; initchrg.f90 uses it in a spin-polarized run only.
        polarization = _strict_float(
            species_value(
                "initial_spin_polarization",
                species_index,
                repr(DEFAULT_INITIAL_SPIN_POLARIZATION),
            ),
            label=f"{symbol} Initial_Spin_Polarization",
        )
        # usrinputfile.F90 reads Atomic_Radius (bohr, 0 by default) for each
        # type in turn; only the moments of a non-collinear run read it.
        atomic_radius = _first_float(
            species_value("atomic_radius", species_index, "0"),
            label=f"{symbol} Atomic_Radius",
        )
        # usrinputfile.F90: p_pot%uu(3,ity), p_pot%jj(3,ity) (d channel).
        hubbard_u = _esdf_energy_rydberg(
            species_value("ldaplusu_u", species_index, "0"),
            label=f"{symbol} ldaplusu_u",
        )
        hubbard_j = _esdf_energy_rydberg(
            species_value("ldaplusu_j", species_index, "0"),
            label=f"{symbol} ldaplusu_j",
        )
        if (hubbard_u != 0.0 or hubbard_j != 0.0) and not _has_d_channel(
            potential_path, format_input
        ):
            # upot.F90 visits the channels of the file only: a U of a species
            # without a d pseudo-wavefunction is never used.
            hubbard_notes.append(
                f"Atom_Type {symbol}: ldaplusu_u/ldaplusu_j is ignored, as in "
                "PARSEC: the pseudopotential has no d channel"
            )
            hubbard_u = hubbard_j = 0.0
        if old_interpolation and not is_periodic:
            format_input["old_interpolation"] = True
        specifications[symbol] = SpeciesPotential(
            potential_path,
            local_l,
            read_valence_density=read_vcd,
            use_spline=use_spline,
            element_symbol=element_symbol,
            atomic_energy_correction=correction,
            spin_orbit=spin_orbit,
            initial_spin_polarization=polarization,
            atomic_radius=atomic_radius,
            hubbard_u=hubbard_u,
            hubbard_j=hubbard_j,
            **format_input,
        )

        species_moments: list[np.ndarray | None] = [None] * len(coordinate_item.value)
        if moment_blocks:
            moment_item = moment_blocks[species_index]
            if len(moment_item.value) != len(coordinate_item.value):
                raise ParsecInputError(
                    f"Atom_Type {symbol}: Initial_NCL_Moment has "
                    f"{len(moment_item.value)} rows but there are "
                    f"{len(coordinate_item.value)} atoms of this type"
                )
            for row_index, moment_line in enumerate(moment_item.value):
                moment_values, moment_unit = _block_numbers(
                    (moment_line,), label=f"{symbol} Initial_NCL_Moment"
                )
                if moment_unit or len(moment_values) != 3:
                    raise ParsecInputError(
                        f"line {moment_item.line}: every Initial_NCL_Moment row "
                        "must contain exactly three numbers"
                    )
                species_moments[row_index] = np.asarray(moment_values, dtype=float)

        # Move_Flag of usrinputfile.F90: 0 (default) moves every atom of the
        # type; n > 0 moves its first n atoms; a negative value reads a
        # fourth column (1 moves, 0 fixes) when the atoms move at all.
        move_flag = _integer(
            species_value("move_flag", species_index, "0"),
            label=f"{symbol} Move_Flag",
        )
        if move_flag > len(coordinate_item.value):
            raise ParsecInputError(
                f"Atom_Type {symbol}: invalid Move_Flag value {move_flag}"
            )
        for row_index, coordinate_line in enumerate(coordinate_item.value):
            values, unit = _block_numbers((coordinate_line,), label=f"{symbol} Atom_Coord")
            wanted = 4 if (move_flag < 0 and atoms_move) else 3
            if unit or len(values) < wanted:
                raise ParsecInputError(
                    f"line {coordinate_item.line}: every Atom_Coord row must "
                    f"contain {'three coordinates and a movement index' if wanted == 4 else 'exactly three numbers'}"
                )
            if len(values) > 3 and move_flag >= 0 and row_index == 0:
                # PARSEC reads three numbers and leaves the rest of the row.
                coordinate_notes.append(
                    f"Atom_Type {symbol}: the Atom_Coord rows hold more than three "
                    "numbers; without a negative Move_Flag the further columns are "
                    "ignored, as in PARSEC."
                )
            if move_flag == 0 or not atoms_move:
                moves = True
            elif move_flag > 0:
                moves = row_index < move_flag
            else:
                if values[3] not in (0.0, 1.0):
                    raise ParsecInputError(
                        f"Atom_Type {symbol}: invalid atom movement index "
                        f"{values[3]:g}; use 0 or 1"
                    )
                moves = values[3] == 1.0
            if fractional_coordinates and andrew_slab:
                # Andrew Grieder's slab: in-plane fractional coordinates folded into the cell, and z the
                # Cartesian coordinate in bohr, as PARSEC reads it (qian/slab_boundary multiplied z by the
                # height 2R of the box, which is no lattice vector: corrected here)
                position = np.asarray(values[:3], dtype=float) * coordinate_factor
                fraction = position[:2] - np.floor(position[:2])
                position = np.array([*(fraction @ periodic_cell.lattice_vectors[:2, :2]), position[2]])
            elif fractional_coordinates:
                # Andrew Grieder's routines: fractional coordinates are folded into the cell
                position = np.asarray(values[:3], dtype=float) * coordinate_factor
                fraction = position - np.floor(position)
                position = fraction @ periodic_cell.lattice_vectors
            elif coordinate_matrix is None:
                position = np.asarray(values[:3], dtype=float) * coordinate_factor
            else:
                position = np.asarray(values[:3], dtype=float) @ coordinate_matrix
            for copy in (
                replicate_positions(position, primitive_vectors, super_cell)
                if replicate
                else (position,)
            ):
                movable.append(moves)
                atoms.append(
                    Atom(
                        symbol,
                        copy,
                        initial_moment=species_moments[row_index],
                    )
                )
        if not any(atom.symbol == symbol for atom in atoms):
            raise ParsecInputError(f"Atom_Type {symbol} has an empty Atom_Coord block")
    if is_periodic and not wire_or_slab and not andrew_slab and not routines.fortran:
        from ..SCF.pbc import atom_outside_cell

        outside = atom_outside_cell(periodic_cell.lattice_vectors, atoms)
        if outside is not None:
            # Andrew Grieder's preparation takes the atoms inside the cell only
            fall_back(f"atom {outside.symbol} at {np.asarray(outside.position).tolist()} outside the cell")
    only_martins = not any(name in {"martins_wang", "fhipp"} for name in formats)
    for key, table in (
        ("core_cutoff_radius", scalar),
        ("electron_per_orbital", blocks),
        ("nonlinear_core_correction", scalar),
        ("local_component", scalar),
    ):
        # PARSEC reads these for some formats only, in their order: a line
        # under another species would go to the next species that reads it.
        readers = FORMATS_READING[key]
        count = sum(name in readers for name in formats)
        if len(table.get(key, [])) <= count:
            continue
        if key == "local_component":
            raise ParsecInputError(
                f"Local_Component occurs {len(table[key])} times, but only "
                f"{count} Atom_Type(s) read it: PARSEC reads none for a "
                "martins_wang species, whose file names its local channel, and "
                "gives the n-th Local_Component to the n-th species that reads "
                "it; remove the lines of the martins_wang species"
            )
        if only_martins and key != "nonlinear_core_correction":
            raise ParsecInputError(
                f"{key} occurs {len(table[key])} times, but only "
                f"{count} Atom_Type(s) have "
                "Pseudopotential_Format: martins_old, the format that reads it"
            )
        names = " or ".join(sorted(readers))
        raise ParsecInputError(
            f"{key} occurs {len(table[key])} times, but only {count} "
            f"Atom_Type(s) have Pseudopotential_Format {names}, the "
            f"format{'s' if len(readers) > 1 else ''} that read{'' if len(readers) > 1 else 's'} "
            "it; PARSEC gives the n-th occurrence to the n-th species that reads it"
        )
    if any(value != 0.0 for value in filter_values):
        # usrinputfile.F90:1897-1907 reads them only for a periodic system.
        coordinate_notes.append(
            "Alpha_Filter, Beta1_Filter and Core_Filter are read by PARSEC for "
            "periodic systems only; they are ignored for this cluster"
        )
    if old_interpolation and is_periodic:
        # Only the cluster builders ionpot.f90 and forcloc.f90 read it.
        coordinate_notes.append(
            "Old_Interpolation_Format applies to clusters only; PARSEC ignores "
            "it for a periodic system"
        )
    if (
        any(item.read_valence_density for item in specifications.values())
        and specifications[symbols[-1]].pseudopotential_format != "martins_new"
    ):
        # pseudo.f90 tests the format of the last Atom_Type, whichever
        # species has Read_VCD, and stops with error 23.
        raise ParsecInputError(
            "Read_VCD is set and the last Atom_Type has a "
            f"{specifications[symbols[-1]].pseudopotential_format} "
            "pseudopotential: PARSEC stops there, because it tests the format "
            "of the last Atom_Type; put a martins_new Atom_Type last"
        )

    if (
        is_periodic
        and any(specification.spin_orbit for specification in specifications.values())
        and not self_consistent_spin_orbit
        and not noncollinear
    ):
        raise UnsupportedParsecOptionError(
            "SO_PSP=true with a periodic Boundary_Conditions is only supported "
            "together with SO_from_scratch/SCF_SO=true (self-consistent SOC, "
            "run_self_consistent_soc_kpoints); the default perturbative "
            "spin-orbit correction does not yet handle periodic k-point "
            "sampling"
        )
    # Double_Grid_Order > 1 with SO_PSP: the spin-orbit projectors are averaged
    # over the double grid as well (nonloc.F90; Eigensolvers/perturbative_soc.py),
    # and so are the LDA+U projectors (upot.F90; Hamiltonian/hubbard.py).

    warnings: list[str] = (
        list(coordinate_notes) + super_cell_notes + warnings_from_kpoints + kpoint_notes + grid_notes
    )
    # PARSEC 1.5 keywords of the input layer (_input15_keywords, Correlation_Type wb).
    warnings.extend(input15_notes)
    warnings.extend(bak15_notes)
    # --esdf-lenient: what it ignored; duplicates found later are added at the end.
    lenient_reported = len(lenient_notes)
    warnings.extend(lenient_notes)
    # PARSEC 1.5 Recenter_Atoms_In_Aperiodic_Directions (default .true.,
    # usrinputfile.F90:3653-3657): .false. keeps the coordinates of the input
    # (no re-centring of a cluster, or of a slab along z; initial.f90:196-327).
    recenter_aperiodic = optional_bool("recenter_atoms_in_aperiodic_directions", True)
    # initial.f90:196-327 of PARSEC 1.5: in every run other than manual
    # movements PARSEC re-centres a slab along z (|z_mov| > 0.1 bohr) before
    # the symmetry search, a restart (Restart_Run) included (1.5 commented
    # out the test istart == 0 of the previous tree, which re-centred first
    # runs only); every later step, force table and output file is in that
    # frame, and a restart from the parsec.dat of a first run with the same
    # input lands in the frame of that run.  Wires are not re-centred.
    slab_shift = 0.0
    if wire_or_slab and periodicity == 2 and minimization != "manual" and recenter_aperiodic:
        from ..V_ion import center_slab_geometry

        placed, would_shift = center_slab_geometry(atoms)
        atoms = list(placed)
        slab_shift = would_shift
    atoms_outside_cell = None
    if is_periodic and routines.fortran and recenter_aperiodic:
        # initial.f90:322-326 (PARSEC 1.5; skipped with Recenter_Atoms_In_Aperiodic_Directions .false.,
        # initial.f90:202): every periodic input is translated by lattice vectors to fractional
        # coordinates in [-1/2, 1/2) along its periodic directions at start (back_to_cell); the previous tree did
        # so after moves only.  An atom inside keeps its coordinates; the periodic sums are unchanged (the
        # preparation translates into [0, 1) for its image bounds, the same positions bit for bit in a cell of
        # orthogonal lattice vectors), and the coordinates printed and moved are those of 1.5.  Andrew Grieder's
        # routines keep the input's.
        from ..Grid.cell import back_to_cell, wrap_into_cell

        atoms_outside_cell = any(
            wrap_into_cell(atom.position, periodic_cell.lattice_vectors) is not atom.position
            for atom in atoms
        ) if periodicity == 3 else None
        atoms = [
            atom
            if (moved := back_to_cell(atom.position, periodic_cell.lattice_vectors, periodicity)) is atom.position
            else replace(atom, position=moved)
            for atom in atoms
        ]
    if old_pseudopotential_format:
        warnings.append(
            "Old_Pseudopotential_Format is not a PARSEC keyword and is ignored; "
            "the format is chosen per species by Pseudopotential_Format "
            "(martins_old for *_POTR.DAT files)"
        )
    # Polarizability with spin-orbit coupling or non-collinear magnetism: as
    # PARSEC (polar.f90 keeps vold and the wavefunctions), every field loop
    # after the first starts from the loop before (its last mixed potential,
    # and the spinors of the isolated spin-orbit loops; cli.py).
    if not polarizability and "electric_field" in scalar and electric_field != 0.0:
        warnings.append(
            "Electric_Field is ignored without Polarizability: .true., as in "
            "PARSEC, which applies a field only in the seven polarizability runs"
        )
    if "atomic_radius" in scalar and not noncollinear:
        warnings.append(
            "Atomic_Radius is read only by the atomic moments of a "
            "non-collinear calculation; here it changes nothing"
        )
    point_charges, point_charge_type_counts = _point_charges(
        scalar, blocks, optional_bool("add_point_charges"), is_periodic,
        coordinate_factor, warnings,
    )
    charged_sheets = _charged_sheets(
        scalar,
        blocks,
        optional_bool("add_charged_sheets"),
        periodicity,
        coordinate_matrix,
        coordinate_factor,
        warnings,
    )
    # External_Mag_Field and Orb_Mag are inert in PARSEC: usrinputfile.F90
    # stores them, and with a field forces complex wavefunctions, but no
    # Zeeman term or vector potential is applied anywhere.
    magnetic_field = (
        _first_float(one("external_mag_field"), label="External_Mag_Field")
        if "external_mag_field" in scalar
        else 0.0
    )
    if magnetic_field != 0.0:
        warnings.append(
            "External_Mag_Field is accepted as in PARSEC, where it applies no "
            "field to the electrons (no Zeeman term or vector potential); the "
            "calculation is that without it"
        )
    if optional_bool("orb_mag"):
        warnings.append(
            "Orb_Mag is accepted as in PARSEC, where it changes nothing of the "
            "calculation"
        )
    complex_wavefunctions = optional_bool("complex_wfn") or magnetic_field != 0.0
    if complex_wavefunctions and not is_periodic:
        warnings.append(
            "Complex wavefunctions (Complex_Wfn or External_Mag_Field): the "
            "Hamiltonian of an isolated system is real, and this code "
            "computes its states as real vectors, which are the same states"
        )
    warnings.extend(hubbard_notes)
    if "total_atom_num" in scalar:
        declared_atoms = _integer(
            one("total_atom_num"), label="Total_Atom_Num"
        )
        if declared_atoms != len(atoms):
            raise ParsecInputError(
                f"Total_Atom_Num={declared_atoms}, but {len(atoms)} coordinates "
                "were found"
            )
    skip_force = optional_bool("skip_force")
    single_point_forces = _small15_ionic_keywords(
        scalar, optional_bool, atoms_move=atoms_move, warnings=warnings
    )
    movement_count = None
    if "movement_num" in scalar:
        movement_count = _integer(one("movement_num"), label="Movement_Num")
        if movement_count < 0:
            raise ParsecInputError("Movement_Num must be nonnegative")
    # Ion_Energy_Diff is not one of them: it is the Casida-Salahub parameter
    # (read with Correlation_Type above).
    ionic_control_keys = {
        "movement_num",
        "force_min",
        "max_step",
        "min_step",
    }
    ionic_controls_present = bool(ionic_control_keys.intersection(scalar))
    force_minimum = (
        _force_rydberg_per_bohr(one("force_min"), label="Force_Min")
        if "force_min" in scalar
        else 1.0e-2
    )
    maximum_step = (
        _physical_length(one("max_step"), label="Max_Step")
        if "max_step" in scalar
        else DEFAULT_RELAXATION_MAXIMUM_STEP
    )
    minimum_step = (
        _physical_length(one("min_step"), label="Min_Step")
        if "min_step" in scalar
        else 1.0e-4
    )
    states = _integer(one("states_num"), label="States_Num")

    eigensolver_method = _normalize_label(one("eigensolver", "chebdav"))
    if eigensolver_method == "arpk":
        eigensolver_method = "arpack"
    if eigensolver_method in {"trlanc", "trlan"}:
        raise UnsupportedParsecOptionError(
            "Eigensolver=trlanc: the thick-restart Lanczos solver was removed "
            "from PARSEC (usrinputfile.F90 ierr 126) and has no source to port"
        )
    if eigensolver_method not in {"chebff", "chebdav", "arpack", "diagla", "none"}:
        raise UnsupportedParsecOptionError(
            f"Eigensolver={eigensolver_method!r} is not supported"
        )
    if eigensolver_method == "diagla":
        warnings.append(
            "Eigensolver: diagla -- the present Fortran PARSEC refuses it in "
            "its input reader (usrinputfile.F90 ierr 126, 'DIAGLA eigensolver "
            "is to be removed from PARSEC'); PARSEC.py runs the DIAGLA block "
            "Davidson solver of diagla.f90z, as earlier PARSEC versions did."
        )
        if is_periodic:
            warnings.append(
                "Eigensolver: diagla on a periodic grid: its preconditioner "
                "(fdo5scl) annihilates the alternating grid mode when every "
                "axis has an even number of points, and DIAGLA can stall once "
                "the residual is that mode (an odd number of points per axis "
                "avoids it)."
            )

    # usrinputfile.F90 reads Chebdav_Degree for CHEBFF and CHEBDAV only.
    first_filter_degree = 20
    if eigensolver_method in {"chebff", "chebdav"}:
        first_filter_degree = _integer(
            one("chebdav_degree", "20"), label="Chebdav_Degree"
        )
    elif "chebdav_degree" in scalar:
        warnings.append(
            f"Chebdav_Degree is not read for Eigensolver={eigensolver_method} "
            "(usrinputfile.F90 reads it for chebff and chebdav only)."
        )
    if eigensolver_method == "chebdav" and first_filter_degree < 15:
        raise ParsecInputError(
            "Chebdav_Degree must be at least 15 when Eigensolver=chebdav"
        )
    if eigensolver_method == "chebff" and first_filter_degree < 10:
        # usrinputfile.F90:2279-2288 of PARSEC 1.5 (the earlier PARSEC: 15).
        warnings.append(
            f"Chebdav_Degree={first_filter_degree} is below 10; "
            "PARSEC 1.5 resets it to 12 (the earlier PARSEC to 15)."
        )
        first_filter_degree = 12

    # Spinor runs (Non_Collinear_magnetism, SO_from_scratch/SCF_SO, a species' SO_PSP) take FF_MaxIter 3 as
    # every run, and keep the degree decrease after iteration 5 (the default since 2026-11-05, checked on
    # A100: with the decrease after 4 the warm-started field loops of the spin-orbit polarizability tests
    # need as many steps as the first one, 31; everything else passes with 3/4 and 3/5).  An earlier
    # FF_MaxIter 2 of spinor runs was needed only while recentring a cluster dropped the moments of
    # Initial_NCL_Moment (fixed on 2026-11-05).
    spinor_run = bool(
        noncollinear
        or self_consistent_spin_orbit
        or any(specification.spin_orbit for specification in specifications.values())
    )
    default_cycles = 3
    if eigensolver_method == "chebff":
        # The default 3 of PARSEC 1.5 (usrinputfile.F90:2242; the earlier
        # PARSEC and this port before 2026-11-04: 2), the default since
        # 2026-11-04 (on A100 the SCF of the nanodiamonds k06-k14 is 7-13 %
        # shorter than with 4, with the same number of SCF steps or one fewer).
        first_filter_cycles = _integer(
            one("ff_maxiter", str(default_cycles)), label="FF_MaxIter"
        )
        # usrinputfile.F90:2303-2310 of PARSEC 1.5: 0..50 (the earlier PARSEC
        # 1..9); a value outside resets to the default 3.
        if first_filter_cycles == 0:
            # PARSEC 1.5 accepts 0 ("to enable testing", usrinputfile.F90:2300-2303) and then
            # runs no first-filter cycle (chebff.f90z:377): after a restart that holds the
            # wavefunctions (restart.h5 of its TierV restart_run_just_check_force tests: chebff
            # in "cont" mode) it keeps them, otherwise it keeps random vectors.
            if restart_run and restart_from_wfndat:
                warnings.append(
                    "FF_MaxIter=0: PARSEC 1.5 runs no first-filter cycle and keeps the stored "
                    "wavefunctions; here the first solve filters the stored states of wfn.dat, "
                    "which FF_MaxIter does not control."
                )
            else:
                warnings.append(
                    "FF_MaxIter=0: PARSEC 1.5 runs no first-filter cycle then; it keeps the "
                    "wavefunctions of a restart.h5 (HDF5), which this port neither writes nor "
                    f"reads, and random vectors otherwise.  The first solve runs the default {default_cycles} cycles."
                )
            first_filter_cycles = default_cycles
        if not 1 <= first_filter_cycles <= 50:
            warnings.append(
                f"FF_MaxIter={first_filter_cycles} is outside 0..50; "
                f"PARSEC 1.5 and PARSEC.py reset it to the default {default_cycles}."
            )
            first_filter_cycles = default_cycles
    else:
        # usrinputfile.F90 reads FF_MaxIter only inside the CHEBFF case.  The
        # value is not a control of CHEBDAV, ARPACK or DIAGLA; it stays 2 for
        # them because the CHEBDAV emulation of the device k-point workers
        # (acceleration/Eigensolvers/chebff_spinor.py) takes it as its least
        # number of cycles, whose results stay those of before 2026-11-04.
        first_filter_cycles = 2

    # usrinputfile.F90: min(nstate/2, 4) for ARPACK, min(nstate/2, 6) for
    # CHEBFF/CHEBDAV; DIAGLA (refused there) takes the latter.
    block_limit = 4 if eigensolver_method == "arpack" else 6
    default_block_size = max(1, min(states // 2, block_limit))
    matvec_block_size = _integer(
        one("matvec_blocksize", str(default_block_size)),
        label="Matvec_Blocksize",
    )
    if matvec_block_size < 1:
        raise ParsecInputError("Matvec_Blocksize must be positive")
    if eigensolver_method == "arpack" and matvec_block_size > states:
        # usrinputfile.F90:2253-2263 of PARSEC 1.5.
        warnings.append(
            f"Matvec_Blocksize={matvec_block_size} is bigger than the number of states; "
            f"PARSEC 1.5 adjusts it to States_Num/2 = {max(1, states // 2)} for ARPACK."
        )
        matvec_block_size = max(1, states // 2)

    filter_degree = _integer(
        one("chebyshev_degree", "15"), label="Chebyshev_Degree"
    )
    default_delta = "1" if filter_degree < 10 else "3"
    filter_delta = _integer(
        one("chebyshev_degree_delta", default_delta),
        label="Chebyshev_Degree_Delta",
    )
    subspace_buffer = _integer(
        one("subspace_buffer_size", "6"), label="Subspace_Buffer_Size"
    )
    # Chebyshev_filtering (solver%do_subsp): forced on for CHEBFF and CHEBDAV
    # (usrinputfile.F90:1452-1455), default off for ARPACK and DIAGLA.
    chebyshev_filtering = None
    if eigensolver_method in {"arpack", "diagla"}:
        chebyshev_filtering = optional_bool("chebyshev_filtering")
    elif (
        eigensolver_method in {"chebff", "chebdav"}
        and "chebyshev_filtering" in scalar
        and not optional_bool("chebyshev_filtering")
    ):
        warnings.append(
            f"Chebyshev_Filtering=false is ignored: PARSEC always runs the "
            f"Chebyshev subspace filter after {eigensolver_method}."
        )
    filtering = eigensolver_method in {"chebff", "chebdav"} or bool(
        chebyshev_filtering
    )
    if not filtering and eigensolver_method != "none":
        unread = [
            label
            for key, label in (
                ("chebyshev_degree", "Chebyshev_Degree"),
                ("chebyshev_degree_delta", "Chebyshev_Degree_Delta"),
            )
            if key in scalar
        ]
        if unread:
            warnings.append(
                f"{' and '.join(unread)} not read: Eigensolver="
                f"{eigensolver_method} without Chebyshev_Filtering runs no "
                "Chebyshev subspace filter."
            )
    # Subspace_Buffer_Size is a keyword of the earlier PARSEC only: PARSEC 1.5
    # does not read it (esdf_key_mod.f90:290 and usrinputfile.F90:5567 of 1.5
    # commented out), and 0..5 reset to 6 as before.  PARSEC 1.5 has no buffer:
    # its working space is exactly States_Num vectors (eigen_solver_module.F90:53,
    # eigval.F90:1344-1356 of 1.5), which the PARSEC.py extension
    # Exact_State_Count: .true. selects.
    exact_requested = "exact_state_count" in scalar and optional_bool("exact_state_count")
    exact_state_count = filtering and exact_requested
    if exact_requested and not filtering:
        warnings.append(
            "Exact_State_Count (PARSEC 1.5's working space of exactly States_Num vectors) is "
            f"read with Chebyshev filtering only; Eigensolver={eigensolver_method} without "
            "Chebyshev_Filtering: no effect here."
        )
    if exact_state_count:
        if "subspace_buffer_size" in scalar:
            warnings.append(
                f"Subspace_Buffer_Size={subspace_buffer} is not used: Exact_State_Count works "
                "with exactly States_Num vectors, as PARSEC 1.5 does."
            )
        subspace_buffer = 0
    elif filtering and subspace_buffer < 6:
        warnings.append(
            f"Subspace_Buffer_Size={subspace_buffer} is below 6 for "
            f"{eigensolver_method} filtering; PARSEC resets it to 6."
        )
        subspace_buffer = 6
    maximum_matvec = _integer(
        one("maximum_matvec", "30000"), label="Maximum_Matvec"
    )
    if maximum_matvec < 1:
        raise ParsecInputError("Maximum_Matvec must be positive")
    # usrinputfile.F90:1444-1449 / parsec.F90:1046 / eigval.F90:679-685:
    # without initeigval the number of eigenpairs per representation is that
    # of the restart file (restart_run.F90, restart.F90), kept in every solve.
    fix_number_eigenvalues = optional_bool("fix_number_eigenvalues")
    if dynamic_tolerance_off_for_restart:
        warnings.append(
            "Dynamic_Diag_Tol is not used with Restart_Run: PARSEC switches it off "
            "(usrinputfile.F90: dynamic tolerance is not working with restart run)"
        )
    if fix_number_eigenvalues and not restart_run:
        # PARSEC then has no number at all (nn = -1 of init_var.F90): its
        # first diagonalization fails (eigval error flag -3, segmentation fault).
        raise UnsupportedParsecOptionError(
            "Fix_Number_Eigenvalues=true keeps the number of eigenvalues of the "
            "restart file; it needs Restart_Run (PARSEC's first diagonalization "
            "fails without one)"
        )
    communication = [
        label
        for key, label in (
            ("exp_comm", "Exp_Comm"),
            ("max_lap_buffers", "Max_Lap_Buffers"),
            # usrinputfile.F90 l.2863: the number of processor groups.
            ("mpi_groups", "MPI_Groups"),
        )
        if key in scalar
    ]
    if communication:
        if "max_lap_buffers" in scalar:
            _integer(one("max_lap_buffers"), label="Max_Lap_Buffers")
        if "mpi_groups" in scalar:
            _integer(one("mpi_groups"), label="MPI_Groups")
        if "exp_comm" in scalar:
            optional_bool("exp_comm")
        warnings.append(
            f"{' and '.join(communication)} tune the MPI communication of "
            "PARSEC's matrix-vector product and have no effect here."
        )

    # The labels of PARSEC 1.5 (parsec_python.Input.solver_options).
    solver_options = read_solver_options(
        scalar,
        one,
        optional_bool,
        integer=_integer,
        real=_strict_float,
        eigensolver_method=eigensolver_method,
        filtering=filtering,
        ionic_motion=(
            _normalize_label(one("minimization", "none")) not in {"none", ""}
            or optional_bool("molecular_dynamics")
        ),
        warnings=warnings,
    )
    # The SUBSPACE lower-bound rule of a periodic cell whose input has no
    # Smart_Subspace line (DEFAULT_PERIODIC_SUBSPACE_LOWER_BOUND, by periodicity).
    periodic_rule = DEFAULT_PERIODIC_SUBSPACE_LOWER_BOUND.get(periodicity, "monotone")
    if (
        filtering
        and "smart_subspace" not in scalar
        and "subspace_lower_bound" not in scalar
        and periodic_rule != "monotone"
    ):
        solver_options.eigensolver["subspace_lower_bound"] = periodic_rule
    # The upper bound of a periodic cell whose input has no Bounds_Method or
    # Bounds_Iter line (DEFAULT_PERIODIC_LANCZOS_BOUNDS, by periodicity).
    periodic_bounds = DEFAULT_PERIODIC_LANCZOS_BOUNDS.get(periodicity, "parsec")
    if (
        filtering
        and "bounds_method" not in scalar
        and "bounds_iter" not in scalar
        and periodic_bounds != "parsec"
    ):
        solver_options.eigensolver["lanczos_bounds"] = periodic_bounds
    if solver_options.filter_restart is not None:
        # parsec_python.SCF.filter_restart, applied by the ionic loops.
        solver_options.eigensolver["filter_restart"] = solver_options.filter_restart
    diagonalization_tolerance = _strict_float(
        one("diag_tolerance", "1e-4"), label="Diag_Tolerance"
    )
    if eigensolver_method == "chebff" and diagonalization_tolerance <= 0.0:
        # usrinputfile.F90:2609-2618 of PARSEC 1.5: -1, its default for
        # CHEBFF, runs FF_MaxIter cycles without a residual test, as CHEBFF
        # always does here; neither CHEBFF nor SUBSPACE reads the tolerance.
        if solver_options.eigensolver.pop("first_filter_residual_test", "off") != "off":
            warnings.append(
                f"FF_Residual_Test with Diag_Tolerance={diagonalization_tolerance:g}: "
                "PARSEC 1.5 tests no residual then (tol > 0 only); CHEBFF runs "
                "FF_MaxIter cycles"
            )
        diagonalization_tolerance = 1.0e-4
    eigensolver = EigensolverSettings(
        method=eigensolver_method,
        tolerance=diagonalization_tolerance,
        first_filter_degree=first_filter_degree,
        first_filter_cycles=first_filter_cycles,
        matvec_block_size=matvec_block_size,
        subspace_buffer=subspace_buffer,
        filter_degree=filter_degree,
        filter_degree_delta=filter_delta,
        chebyshev_filtering=chebyshev_filtering,
        dynamic_tolerance=dynamic_tolerance,
        maximum_matvec=maximum_matvec,
        fixed_state_count=fix_number_eigenvalues,
        exact_state_count=exact_state_count,
        **solver_options.eigensolver,
    )

    # usrinputfile.F90:1540-1600: the mixer and its keywords, every one read
    # with its PARSEC default whatever the method.
    mixing_name = _normalize_label(one("mixing_method", "anderson"))
    if mixing_name not in MIXING_METHODS:
        raise ParsecInputError(
            f"Mixing_Method: unknown mixer type {mixing_name!r}; PARSEC's "
            "mixers are " + ", ".join(MIXING_METHODS)
        )
    safeguard = optional_bool("mixing_safeguard")
    if safeguard and mixing_name != "anderson":
        raise UnsupportedParsecOptionError(
            f"Mixing_Safeguard is an option of the Anderson mixer of "
            f"PARSEC.py; Mixing_Method={mixing_name!r} has none"
        )
    mixing = MixingSettings(
        parameter=_strict_float(one("mixing_param", "0.30"), label="Mixing_Param"),
        memory=_integer(one("memory_param", "4"), label="Memory_Param"),
        restart=_integer(one("restart_mixing", "20"), label="Restart_Mixing"),
        safeguard=safeguard,
        step_limit=_strict_float(
            one("mixing_step_limit", "2.0"), label="Mixing_Step_Limit"
        ),
        growth_trigger=_strict_float(
            one("mixing_growth_trigger", "2.0"),
            label="Mixing_Growth_Trigger",
        ),
        backoff=_strict_float(
            one("mixing_backoff", "0.5"), label="Mixing_Backoff"
        ),
        method=mixing_name,
        group_size=_integer(
            one("mixing_group_size", "1"), label="Mixing_Group_Size"
        ),
        en_like=_integer(one("mixing_en_like", "0"), label="Mixing_EN_Like"),
        restart_factor=_strict_float(
            one("mixing_restart_factor", "0.3"), label="Mixing_Restart_Factor"
        ),
        expand_factor=_strict_float(
            one("mixing_memory_expand_factor", "2.0"),
            label="Mixing_Memory_Expand_Factor",
        ),
        preferred_type=_integer(
            one("mixing_preferred_type", "1"), label="Mixing_Preferred_Type"
        ),
        **_mixing_preconditioner_fields(scalar, one, periodicity=periodicity, warnings=warnings),
    )
    # Keywords PARSEC reads but the selected mixer never uses.
    multisecant_keys = (
        "Mixing_Group_Size",
        "Mixing_EN_Like",
        "Mixing_Restart_Factor",
        "Mixing_Memory_Expand_Factor",
        "Mixing_Preferred_Type",
    )
    inert_mixing_keys = {
        "anderson": multisecant_keys,
        "broyden": multisecant_keys,
        "msecant1": ("Restart_Mixing", "Mixing_Preferred_Type"),
        "msecant2": ("Restart_Mixing", "Mixing_Preferred_Type"),
        "msecant3": ("Restart_Mixing",),
    }[mixing_name]
    for keyword in inert_mixing_keys:
        if keyword.lower() in scalar:
            warnings.append(
                f"{keyword} has no effect with Mixing_Method={mixing_name}; "
                "PARSEC reads it and its mixer does not use it."
            )
    # genbro.f90 keeps its pairs in files of mixer%block - 1 slots and drops
    # some from the steps after the block (see Mixer/broyden.py); this port
    # keeps them all.
    broyden_block = max(20, mixing.memory + 1)
    if mixing_name == "broyden" and mixing.memory > 0 and mixing.restart > broyden_block:
        warnings.append(
            f"Restart_Mixing={mixing.restart} exceeds the Broyden block of "
            f"PARSEC ({broyden_block}): genbro.f90 then drops the pairs of the "
            f"steps 1+k*{broyden_block - 1} (k >= 1) of each restart cycle "
            "from the later steps; PARSEC.py keeps every pair since the last "
            "restart, so the SCF histories differ from that step on."
        )
    # Solver_Lpole is PARSEC's keyword.  The two below are not: they control
    # what a prepared system adds to the multipole boundary values, and
    # "Hartree_Boundary_Tolerance: off" restores PARSEC's boundary.  "auto",
    # and no line beside a radius left to the default rule, is the tolerance
    # that rule derives from the electron count: it replaces the default
    # below.
    boundary_tolerance_text = one("hartree_boundary_tolerance", "1e-3 Ry")
    boundary_tolerance_label = re.sub(
        r"[.\s_-]", "", boundary_tolerance_text.lower()
    )
    default_tolerance = boundary_tolerance_label == "auto" or (
        default_radius and "hartree_boundary_tolerance" not in scalar
    )
    if boundary_tolerance_label == "auto":
        boundary_tolerance_text = "1e-3 Ry"
    if boundary_tolerance_label in {"off", "none", "no", "false"}:
        boundary_tolerance = None
    else:
        boundary_tolerance = _energy_rydberg(
            boundary_tolerance_text, label="Hartree_Boundary_Tolerance"
        )
    atomic_tail_text = re.sub(
        r"[.\s_-]", "", one("hartree_atomic_tail", "auto").lower()
    )
    if atomic_tail_text == "auto":
        atomic_tail = "auto"
    else:
        atomic_tail = (
            "on"
            if _boolean(atomic_tail_text, label="Hartree_Atomic_Tail")
            else "off"
        )
    try:
        hartree = HartreeSettings(
            multipole_order=_integer(
                one("solver_lpole", "9"), label="Solver_Lpole"
            ),
            boundary_method="direct" if optional_bool("full_hartree") else "auto",
            boundary_tolerance=boundary_tolerance,
            atomic_tail=atomic_tail,
            # Hartree_CG_Tolerance / Hartree_CG_Criterion of PARSEC 1.5 (_input15_keywords)
            **input15.hartree_controls,
        )
    except ValueError as error:
        if isinstance(error, ParsecInputError):
            raise
        raise ParsecInputError(
            f"Solver_Lpole/Hartree_Boundary_Tolerance: {error}"
        ) from error
    fermi_temperature = _temperature_kelvin(
        one("fermi_temp", "80"), label="Fermi_Temp"
    )
    fixed_occupations = None
    fixed_occupations_down = None
    if fermi_temperature < 0 and spin_polarized:
        # initial.f90 reads one section per spin channel, the up one first.
        from ..Occupations import read_spin_occupation_file

        spin_file = read_spin_occupation_file(source.parent / "occup.in", states)
        fixed_occupations = spin_file.occupations_up
        fixed_occupations_down = spin_file.occupations_down
        if spin_file.unread_lines:
            warnings.append(
                f"{spin_file.source.name}: {spin_file.unread_lines} line(s) "
                "after the sections of the two spin channels are ignored."
            )
    elif fermi_temperature < 0:
        # initial.f90: a negative temperature reads the occupations from the
        # file occup.in of the run directory.
        from ..Occupations import read_occupation_file

        occupation_file = read_occupation_file(source.parent / "occup.in", states)
        fixed_occupations = occupation_file.occupations
        if occupation_file.unread_lines:
            warnings.append(
                f"{occupation_file.source.name}: only the first section is "
                "read, as PARSEC does without spin polarization; "
                f"{occupation_file.unread_lines} further line(s), the "
                "sections of other spin channels, are ignored."
            )
    # usrinputfile.F90, "VDW PART": the TS correction needs PBE (ierr 158) and
    # no symmetry (ierr 159); a periodic system must be bulk (ierr 160, the
    # only periodic case here) and reads the Hirshfeld cells Hf_One/Two/Three.
    van_der_waals = optional_bool("van_der_waals")
    hirshfeld_cells = (0, 0, 0)
    hirshfeld_keys = [key for key in ("hf_one", "hf_two", "hf_three") if key in scalar]
    if van_der_waals:
        # usrinputfile.F90:4100: (icorr /= 'pb') .or. use_libxc -> ierr 158 (the TS
        # damping s_R is PBE's, and Correlation_Type does not name a libxc functional).
        if correlation not in {"pb", "pbe"} or input15.libxc is not None:
            raise ParsecInputError(
                "Van_Der_Waals: the TS correction is implemented only for the "
                "GGA-PBE functional of this code (Correlation_Type pb, no libxc functional; "
                "PARSEC error 158)"
            )
        if not ignore_symmetry:
            raise ParsecInputError(
                "Van_Der_Waals: for the TS correction symmetry has to be "
                "switched off (Ignore_Symmetry: true; PARSEC error 159)"
            )
        if is_periodic:
            values = []
            for key, label in (("hf_one", "Hf_One"), ("hf_two", "Hf_Two"), ("hf_three", "Hf_Three")):
                value = _integer(one(key, "0"), label=label)
                if value < 0:
                    raise ParsecInputError(f"{label} must be nonnegative")
                values.append(value)
            hirshfeld_cells = tuple(values)
        elif hirshfeld_keys:
            warnings.append(
                "Hf_One/Hf_Two/Hf_Three are read for a periodic system only "
                "and are ignored, as in PARSEC"
            )
    elif hirshfeld_keys:
        warnings.append(
            "Hf_One/Hf_Two/Hf_Three are present without Van_Der_Waals: true "
            "and are ignored"
        )

    spin_symmetry_sectors = _spin_symmetry_sectors(
        scalar, one, spin_polarized=spin_polarized, warnings=warnings
    )
    scf = SCFSettings(
        max_iterations=_integer(one("max_iter", "50"), label="Max_Iter"),
        convergence_criterion=_energy_rydberg(
            one("convergence_criterion", "2e-4 Ry"),
            label="Convergence_Criterion",
        ),
        fermi_temperature_kelvin=fermi_temperature,
        fixed_occupations=fixed_occupations,
        spin_polarized=spin_polarized,
        fixed_occupations_down=fixed_occupations_down,
        number_of_states=states,
        net_charge=_charge_electrons(
            one("net_charges", "0"), label="Net_Charges"
        ),
        use_plain_residual=optional_bool("use_plain_sre"),
        convergence_criterion_approach=(
            _energy_rydberg(
                one("convergence_criterion_approach"),
                label="Convergence_Criterion_Approach",
            )
            if "convergence_criterion_approach" in scalar
            else None
        ),
        normalize_initial_density=optional_bool(
            "normalize_initial_density", True
        ),
        xc_functional=xc_functional,
        ionization_energy_difference=ionization_energy_difference,
        **({} if exact_exchange is None else {"exact_exchange": exact_exchange}),
        **({} if spin_symmetry_sectors is None else {"spin_symmetry_sectors": spin_symmetry_sectors}),
        self_consistent_spin_orbit=self_consistent_spin_orbit
        or (
            noncollinear
            and any(specification.spin_orbit for specification in specifications.values())
        ),
        noncollinear=noncollinear,
        van_der_waals=van_der_waals,
        hirshfeld_cells=hirshfeld_cells,
        # spinor runs: the degree decrease after iteration 5 unless written (see spinor_run above)
        **{**({"degree_decrease_after": 5} if spinor_run else {}), **solver_options.scf},
    )

    initial_labels = []
    for key in ("initial_density", "density_method"):
        if key in scalar:
            initial_labels.append((key, _normalize_label(one(key))))
    if len(initial_labels) == 2 and initial_labels[0][1] != initial_labels[1][1]:
        raise ParsecInputError(
            "Initial_Density and legacy Density_Method specify different methods"
        )
    initial_method = initial_labels[0][1] if initial_labels else "sad"
    if initial_method == "ml":
        initial_method = "file"
    if initial_method == "sad_ml_grid":
        raise UnsupportedParsecOptionError(
            "Density_Method=sad_ml_grid changed the DFT grid in the old architecture. "
            "The modular implementation keeps Grid_Spacing/domain authoritative; "
            "use Initial_Density=sad, file, charge3net, scdp, or custom."
        )
    file_keys = [key for key in ("ml_density_file", "ml_file_path") if key in scalar]
    if len(file_keys) > 1:
        raise ParsecInputError(
            "provide only one of ML_Density_File and legacy ML_File_Path"
        )
    density_file = input_relative_path(file_keys[0]) if file_keys else None
    model_method = initial_method not in {"sad", "file"}
    default_model = "qm9"
    if model_method:
        from ..MLDensity.registry import get_provider, initial_density_methods

        try:
            default_model = get_provider(initial_method).default_model
        except ValueError:
            raise ParsecInputError(
                f"unknown Initial_Density {initial_method!r}; use one of "
                + ", ".join(initial_density_methods())
            ) from None
    density_cache = input_relative_path("ml_density_cache")
    if density_cache is None and model_method:
        density_cache = source.parent / ".parsec_ml_density_cache"
    initial_density_settings = InitialDensitySettings(
        method=initial_method,
        file=density_file,
        units=normalize_density_units(one("ml_density_units", "auto")),
        negative_policy=_normalize_label(
            one("ml_density_negative_policy", "clip")
        ),
        interpolation=_normalize_label(
            one("ml_density_interpolation", "linear")
        ),
        repository=input_relative_path("ml_density_repository"),
        checkpoint=input_relative_path("ml_density_checkpoint"),
        python_executable=input_relative_path("ml_density_python", follow_links=False),
        model=one("ml_density_model", default_model).strip().lower(),
        device=_normalize_label(one("ml_density_device", "auto")),
        cache_directory=density_cache,
        regenerate=optional_bool("ml_density_regenerate"),
        prediction_chunk_size=_integer(
            one("ml_density_chunk_size", "50000"),
            label="ML_Density_Chunk_Size",
        ),
        bridge=input_relative_path("ml_density_bridge"),
    )

    scf_upscale = 1.0
    if "relaxation_scf_upscale" in scalar:
        scf_upscale = _strict_float(
            one("relaxation_scf_upscale"), label="Relaxation_SCF_Upscale"
        )
        if not (np.isfinite(scf_upscale) and scf_upscale >= 1.0):
            raise ParsecInputError("Relaxation_SCF_Upscale must be a number of at least 1")
        if minimization not in {"bfgs", "simple"}:
            if scf_upscale != 1.0:
                warnings.append(
                    "Relaxation_SCF_Upscale loosens the SCF loops of a BFGS or "
                    "steepest-descent relaxation; here it is ignored."
                )
            scf_upscale = 1.0

    relaxation = None
    use_symmetry = True
    # A manual run keeps the coordinates of the input and of manual.dat as
    # they are; every other calculation recentres the cluster, but for a
    # restart (initial.f90 l.133-134: istart == 0 only).
    recenter = minimization != "manual" and not restart_run and recenter_aperiodic
    # PARSEC 1.5 Write_Relax_Restart (domove.f90:390): 1.5 writes
    # relax_restart.dat only with .true.; this port keeps writing it unless
    # the input says .false.
    write_relax_restart = optional_bool("write_relax_restart", True)
    if "write_relax_restart" in scalar and minimization == "none":
        warnings.append(
            "Write_Relax_Restart applies to a relaxation (Minimization); it changes nothing here"
        )
    if minimization != "none":
        # A periodic cell (also a wire or slab) relaxes with a fixed cell, as
        # in PARSEC; the movements a reduced k-point set cannot follow are
        # refused below (_check_reduced_kpoint_movements).
        relaxation = RelaxationSettings(
            method=minimization,
            max_movements=500 if movement_count is None else movement_count,
            minimum_force=force_minimum,
            maximum_step=maximum_step,
            minimum_step=minimum_step,
            bfgs_corrections=_integer(
                one("bfgs_number_corr", "7"), label="BFGS_number_corr"
            ),
            restart=optional_bool("relax_restart"),
            scf_upscale=scf_upscale,
            write_restart=write_relax_restart,
        )
        if minimization == "manual":
            # parsec.F90 uses no symmetry when the structures are imposed,
            # and usrinputfile.F90 makes every atom movable.
            use_symmetry = False
            movable = [True] * len(atoms)
        elif not any(movable):
            raise ParsecInputError(
                "Minimization is requested, but Move_Flag leaves no atom movable"
            )
    else:
        if ionic_controls_present:
            warnings.append(
                "Ionic relaxation thresholds/step controls are present without an "
                "active minimizer and are ignored for this fixed single point."
            )
        if optional_bool("relax_restart"):
            warnings.append(
                "Relax_Restart is set without Minimization and is ignored."
            )
    if molecular_dynamics.enabled:
        if not any(movable):
            raise ParsecInputError(
                "Molecular_Dynamics is requested, but Move_Flag leaves no atom movable"
            )
        # parsec.F90 uses no symmetry in molecular dynamics.
        use_symmetry = False
    if polarizability:
        # parsec.F90 uses no symmetry when it applies an electric field.
        use_symmetry = False
    if point_charges:
        # PARSEC finds the point group of the atoms alone (a defect of the
        # Fortran unless Ignore_Symmetry is set); a group the point charges
        # do not have is not used here.
        use_symmetry = False
    if foldblock_side != 1:
        # A domain of whole blocks is no image of itself under the reflections (usrinputfile.F90:814-817
        # switches the symmetry off), also not on request of the command line.
        use_symmetry = False
    symmetrize_density = optional_bool("symmetrize_charge_density")
    space_group = None
    if wire_or_slab:
        # The space group of a wire or slab (and what it decides) is found
        # below, once bandstruc and nscf_kpoints are read.
        pass
    elif is_periodic and not routines.fortran:
        # Andrew Grieder's routines put no fractional translations in the grid.  The Monkhorst_Pack_Shift mesh of a
        # bulk single point of scalar orbitals is reduced by the operations of the crystal that map his grid onto
        # itself, with time reversal, and the density is symmetrized with them (_default_routines_wedge); a hybrid
        # (its exchange sums over the full mesh), spinors and moving atoms keep the full list of equal weights.
        keeps_full_mesh = bool(
            scf.is_hybrid
            or noncollinear
            or self_consistent_spin_orbit
            or any(specification.spin_orbit for specification in specifications.values())
            or atoms_move
            or single_point_forces
        )
        weighted_list = bool(
            explicit_kpoint_weights is not None
            and not np.allclose(explicit_kpoint_weights, explicit_kpoint_weights[0], rtol=1.0e-12, atol=0.0)
        )
        averages = periodicity == 3 and (
            (default_routines_mesh is not None and not keeps_full_mesh)
            or (weighted_list and not (atoms_move or single_point_forces or scf.is_hybrid
                                       or noncollinear or self_consistent_spin_orbit
                                       or any(s.spin_orbit for s in specifications.values())))
        )
        if "symmetrize_charge_density" in scalar and symmetrize_density and not averages:
            # these routines average the density of a reduced k-point set only (a Monkhorst_Pack_Shift wedge, a
            # weighted Kpoint_List of a bulk single point); any other request runs on the Fortran-derived ones, as
            # every Symmetrize_Charge_Density: .true. did before 2026-11-08
            fall_back(
                "Symmetrize_Charge_Density: .true. (the default routines average the density of a Monkhorst_Pack_Shift "
                "wedge or a weighted Kpoint_List of a bulk single point of scalar orbitals only)"
            )
        if default_routines_mesh is not None and not keeps_full_mesh:
            symmetrize_density, space_group = _default_routines_wedge(
                atoms,
                periodic_cell,
                grid,
                default_routines_mesh,
                ignore_symmetry=ignore_symmetry,
                symmetrize_density=symmetrize_density,
                symmetrize_given="symmetrize_charge_density" in scalar,
                warnings=warnings,
            )
            monkhorst_pack_dimensions, monkhorst_pack_shift = default_routines_mesh
            explicit_kpoint_array = None
            explicit_kpoint_weights = None
        if weighted_list and periodicity == 3 and not (atoms_move or single_point_forces):
            # A Kpoint_List of unequal weights runs as given; Symmetrize_Charge_Density averages its density over
            # the operations that map the grid onto itself (an irreducible wedge; scalar orbitals, no hybrid: any
            # other request went to the Fortran-derived routines above).
            symmetrize_density, space_group = _default_routines_list_symmetry(
                atoms,
                periodic_cell,
                grid,
                ignore_symmetry=ignore_symmetry,
                symmetrize_density=symmetrize_density,
                warnings=warnings,
            )
    elif is_periodic:
        weighted_kpoints = bool(
            explicit_kpoint_weights is not None
            and not np.allclose(explicit_kpoint_weights, explicit_kpoint_weights[0])
        )
        forces_requested = atoms_move or single_point_forces
        symmetrize_density, space_group, grid = _periodic_symmetry(
            atoms,
            periodic_cell,
            grid,
            general_periodic_grid=general_periodic_grid,
            ignore_symmetry=ignore_symmetry,
            monkhorst_pack_shift=monkhorst_pack_shift,
            symmetrize_density=symmetrize_density,
            symmetrize_given="symmetrize_charge_density" in scalar,
            noncollinear=noncollinear,
            warnings=warnings,
            forces_of_weighted_kpoints=forces_requested and weighted_kpoints,
            search_info=_bulk_symmetry_search_info(
                periodic_cell,
                ignore_symmetry=ignore_symmetry,
                kpoints=monkhorst_pack_dimensions is not None or explicit_kpoint_array is not None,
                spinors=bool(
                    noncollinear
                    or self_consistent_spin_orbit
                    or any(specification.spin_orbit for specification in specifications.values())
                ),
                bands=bool(blocks.get("bandstruc")),
                nscf="nscf_kpoints" in blocks,
                molecular_dynamics=molecular_dynamics.enabled,
                manual_movement=minimization == "manual",
                requested_shift=requested_grid_shift,
            ),
        )
        # The space group's fractional translations are in the grid settings now: the counts of adjust_step decide
        # between the orthorhombic grid and Fortran's general grid.
        general_periodic_grid = uses_general_periodic_grid(periodic_cell, grid)
        grid = _fortran_bulk_box_lengths(grid, periodic_cell, general_periodic_grid)
        if general_periodic_grid and double_grid_order > 1:
            raise UnsupportedParsecOptionError(
                "Double_Grid_Order > 1 is not implemented on Fortran's general "
                "periodic grid (non-orthogonal lattice vectors, or steps that "
                "differ between the lattice directions): its projectors use one grid step"
            )
        if forces_requested:
            _check_reduced_kpoint_movements(
                atoms,
                periodic_cell,
                space_group,
                movable=movable,
                relaxation=minimization in {"bfgs", "simple"},
                molecular_dynamics=molecular_dynamics.enabled or minimization == "manual",
                reduced=bool(monkhorst_pack_shift is not None or weighted_kpoints),
                ignore_symmetry=ignore_symmetry,
            )
    elif symmetrize_density:
        # The isolated spin-orbit and non-collinear drivers average their
        # densities like the collinear ones (symm_scalar of every rho(:,isp);
        # non-collinear: rho and |m| as scalars, the direction of m kept).
        if ignore_symmetry or not use_symmetry or shape != "sphere":
            warnings.append(
                "Symmetrize_Charge_Density changes nothing here: the "
                "calculation uses no point group (Ignore_Symmetry, a box domain, "
                "manual movements, molecular dynamics, Polarizability or point "
                "charges), so PARSEC's group has the identity alone"
            )
            symmetrize_density = False
        elif requested_grid_shift is not None and not (
            all(value == 0.0 for value in grid.shift) or all(value == 0.5 for value in grid.shift)
        ):
            # An operation M of the point group maps the points (k + s) h of a cluster onto points only where
            # M s - s is integral (Grid.point_group.grid_point_group, which the run would stop at); PARSEC 1.5
            # averages over the whole group anyway, at the nearest points.
            from ..Forces.symmetry import atom_symmetry_operations
            from ..V_ion import center_cluster_geometry

            shift = np.asarray(grid.shift, dtype=np.float64)
            placed = center_cluster_geometry(atoms) if recenter else tuple(atoms)
            for operation in atom_symmetry_operations(placed):
                offset = operation.matrix @ shift - shift
                if np.max(np.abs(offset - np.rint(offset))) > 1.0e-9:
                    raise UnsupportedParsecOptionError(
                        "Symmetrize_Charge_Density with the Grid_Shift "
                        + " ".join(f"{value:g}" for value in grid.shift)
                        + ": an operation of the point group of the atoms does not map the grid onto itself "
                        "(every operation must take the shift into itself modulo whole steps; a shift of 0 or "
                        "1/2 in every direction always does); set Ignore_Symmetry: true or "
                        "Symmetrize_Charge_Density: false"
                    )
    if symmetrize_density and not wire_or_slab:
        scf = replace(scf, symmetrize_density=True)
    # PARSEC always evaluates the forces.  Here a single point evaluates
    # them only on request, so that inputs without the line keep their cost.
    compute_forces = atoms_move or single_point_forces
    if periodicity == 3 and choice_matters and routines.requested is None:
        # the bulk features whose default is the Fortran-derived routines (Gamma-point forces, a space group
        # that sets Fortran's grid), unless the input says Use_Fortran_Routines
        kpoint_input = monkhorst_pack_dimensions is not None or explicit_kpoint_array is not None
        for feature in _routines_module.bulk_default_features(
            atoms,
            periodic_cell,
            grid,
            kpoints=kpoint_input,
            gamma_only=_gamma_only_sampling(monkhorst_pack_dimensions, explicit_kpoint_array),
            forces=compute_forces,
            spinors=bool(self_consistent_spin_orbit or noncollinear),
            spin_polarized=spin_polarized,
            ignore_symmetry=ignore_symmetry,
            spline_species=all(specification.use_spline for specification in specifications.values()),
            search_info=_bulk_symmetry_search_info(
                periodic_cell,
                ignore_symmetry=ignore_symmetry,
                kpoints=kpoint_input,
                spinors=bool(
                    noncollinear
                    or self_consistent_spin_orbit
                    or any(specification.spin_orbit for specification in specifications.values())
                ),
                bands=bool(blocks.get("bandstruc")),
                nscf="nscf_kpoints" in blocks,
                molecular_dynamics=molecular_dynamics.enabled,
                manual_movement=minimization == "manual",
                requested_shift=requested_grid_shift,
            ),
            requested_shift=requested_grid_shift,
        ):
            fortran_by_default(feature)
    if not molecular_dynamics.enabled and any(
        key in scalar for key in _MOLECULAR_DYNAMICS_KEYS if key != "molecular_dynamics"
    ):
        warnings.append(
            "Molecular dynamics controls are present without "
            "Molecular_Dynamics: true and are ignored."
        )
    if is_periodic and not routines.fortran:
        # Andrew Grieder's routines: a crystal's forces are not symmetrized: PARSEC.py finds no space-group
        # operations (the point-group operations of a cluster do not apply), and the structure is relaxed as it is.
        use_symmetry = False

    potential_extrapolation = "none"
    if "potential_extrapolation" in scalar:
        from ..SCF.single_point import POTENTIAL_EXTRAPOLATIONS

        potential_extrapolation = _normalize_label(one("potential_extrapolation"))
        if potential_extrapolation not in POTENTIAL_EXTRAPOLATIONS:
            raise ParsecInputError(
                "Potential_Extrapolation must be one of "
                + ", ".join(POTENTIAL_EXTRAPOLATIONS)
            )
        if not atoms_move:
            warnings.append(
                "Potential_Extrapolation is set, but no atom moves; it is ignored."
            )

    md_output_interval = 1
    if "md_output_interval" in scalar:
        md_output_interval = _integer(one("md_output_interval"), label="MD_Output_Interval")
        if md_output_interval < 1:
            raise ParsecInputError("MD_Output_Interval must be at least 1")
        if molecular_dynamics is None or not molecular_dynamics.enabled:
            warnings.append("MD_Output_Interval is set, but this is not a molecular dynamics run; it is ignored.")

    mlff = None
    structure_source = _normalize_label(one("initial_structure", "input"))
    if structure_source not in {"input", "mlff"}:
        raise ParsecInputError("Initial_Structure must be input or mlff")
    if structure_source == "mlff":
        if is_periodic:
            raise UnsupportedParsecOptionError(
                "Initial_Structure: mlff with Boundary_Conditions=bulk: the force-field "
                "relaxation is implemented for isolated systems only"
            )
        mlff = _mlff_settings(one, scalar, source)
    elif any(key.startswith("mlff_") for key in scalar):
        warnings.append(
            "MLFF_* keywords are present without Initial_Structure: mlff and are ignored."
        )

    energy_refinement = None
    if "energy_refinement" in scalar:
        from ..Energy.refinement import check_refinement_request, normalize_refinement

        try:
            energy_refinement = normalize_refinement(one("energy_refinement"))
            check_refinement_request(
                energy_refinement,
                periodic=is_periodic,
                molecular_dynamics=molecular_dynamics.enabled,
            )
        except ValueError as error:
            raise UnsupportedParsecOptionError(str(error)) from error

    if ignore_symmetry and not is_periodic:
        warnings.append(
            "Ignore_Symmetry=true reproduces PARSEC's zero grid shift and "
            "disables accelerated symmetry reduction unless --symmetry=on "
            "explicitly overrides it."
        )
    if not is_periodic and shape == "box" and hartree.boundary_method != "direct":
        warnings.append(
            "Box Hartree boundaries use the exact direct Coulomb sum in auto mode "
            "and may be expensive."
        )

    band_settings: BandStructureSettings | None = None
    band_blocks = blocks.get("bandstruc", [])
    # PARSEC 1.5: 1.5 reads Bandstruc_Points with or without a
    # bandstruc block (usrinputfile.F90:1622) and rounds the points per line
    # with nint (calc_bands.f90:152-158); its default became 100.  This port
    # keeps its default, 20 points with the truncation of the previous tree,
    # where the input gives no Bandstruc_Points, and takes 1.5's rounding where
    # it does (since 2026-11-04 the rule for defaults that 1.5 changed).  The new
    # Bands_FF_MaxIter (2243) sets the CHEBFF cycles of the band k-points.
    band_points_given = "bandstruc_points" in scalar
    band_points = _integer(one("bandstruc_points", "20"), label="Bandstruc_Points")
    if band_points < 1:
        raise ParsecInputError("Bandstruc_Points must be at least 1")
    band_filter_cycles = None
    if "bands_ff_maxiter" in scalar:
        band_filter_cycles = _integer(one("bands_ff_maxiter"), label="Bands_FF_MaxIter")
        # (1.5 checks no value; its bak_complete_parsec.in writes FF_MaxIter's, 0 for its test inputs: without a
        # bandstruc block the value is read and ignored, below)
        if band_filter_cycles < 1 and band_blocks:
            raise ParsecInputError("Bands_FF_MaxIter must be at least 1")
    if not band_blocks and ("bandstruc_points" in scalar or "bands_ff_maxiter" in scalar):
        warnings.append(
            "Bandstruc_Points and Bands_FF_MaxIter apply to a bandstruc block; "
            "without one no band structure is computed (PARSEC reads and ignores them)"
        )
    elif band_blocks and band_filter_cycles is not None and eigensolver_method != "chebff":
        warnings.append(
            f"Bands_FF_MaxIter sets the filter cycles of the CHEBFF first solver; "
            f"no effect with Eigensolver {eigensolver_method}"
        )
    if band_blocks:
        if not is_periodic:
            raise ParsecInputError("bandstruc applies only to periodic systems")
        if len(band_blocks) != 1:
            raise ParsecInputError("provide a single bandstruc block")
        segments = []
        for line in band_blocks[0].value:
            tokens = line.replace("\t", " ").split()
            try:
                if len(tokens) < 7:
                    raise ValueError("expected 'n kx0 ky0 kz0 kx1 ky1 kz1 label'")
                int(tokens[0])
                numbers = [float(token.lower().replace("d", "e")) for token in tokens[1:7]]
                segments.append(
                    BandSegment(
                        start=tuple(numbers[:3]),
                        end=tuple(numbers[3:]),
                        name=" ".join(tokens[7:]),
                    )
                )
            except ValueError as error:
                raise ParsecInputError(
                    f"bandstruc line {band_blocks[0].line}: {error}; every line is "
                    "'n kx0 ky0 kz0 kx1 ky1 kz1 label' in reciprocal-lattice coordinates"
                ) from error
        try:
            band_settings = BandStructureSettings(
                segments=tuple(segments),
                points=band_points,
                ff_maxiter=band_filter_cycles,
                rounding="nint" if band_points_given else "truncate",
            )
        except ValueError as error:
            raise ParsecInputError(f"bandstruc: {error}") from error

    domain_energy_tolerance = _energy_rydberg(
        one("domain_energy_tolerance", "1e-3 Ry"), label="Domain_Energy_Tolerance"
    )
    if not domain_energy_tolerance > 0.0:
        raise ParsecInputError("Domain_Energy_Tolerance must be positive")
    multipole_boundary = shape == "sphere" and hartree.boundary_method != "direct"
    if default_radius and multipole_boundary and (
        boundary_tolerance is None or atomic_tail == "off"
    ):
        raise ParsecInputError(
            "Boundary_Sphere_Radius is left to the default rule, and the "
            "default radius is chosen for the controlled Hartree boundary: "
            "with Hartree_Boundary_Tolerance: off or Hartree_Atomic_Tail: off "
            "give Boundary_Sphere_Radius"
        )
    domain = None
    domain_rule_seconds = 0.0
    boundary_tolerance_from = (
        "input" if "hartree_boundary_tolerance" in scalar else "default"
    )
    if default_radius or (default_tolerance and multipole_boundary):
        # The rule reads the geometry as the driver places it and the free
        # atoms of the pseudopotential files.  An input with a radius and
        # without "Hartree_Boundary_Tolerance: auto" does not come here.
        from ..Hartree.domain import auto_boundary_tolerance, default_domain
        from ..V_ion import (
            center_cluster_geometry,
            ionic_charge,
            load_pseudopotentials,
        )

        rule_started = perf_counter()
        placed = center_cluster_geometry(atoms) if recenter else tuple(atoms)
        potentials = load_pseudopotentials(
            specifications, xc_functional=xc_functional
        )
        electrons = ionic_charge(placed, potentials) - scf.net_charge
        if electrons <= 0:
            raise ParsecInputError("the requested system has no valence electrons")
        if default_radius:
            domain = default_domain(
                placed,
                potentials,
                electrons,
                spacing=grid.spacing,
                shift=grid.shift[0],
                stencil_half_width=grid.stencil_half_width,
                budget=domain_energy_tolerance,
                boundary_tolerance=None if default_tolerance else boundary_tolerance,
                multipole_boundary=multipole_boundary,
            )
            # The run takes what the two texts read back to, as an input
            # that holds them does.
            grid = replace(
                grid,
                radius=_physical_length(
                    domain.radius_text, label="Boundary_Sphere_Radius"
                ),
            )
            rule_tolerance_text = domain.tolerance_text
            if domain.extra_electrons > 1.0e-6:
                warnings.append(
                    f"{domain.extra_electrons:g} electron(s) more than the free "
                    "atoms hold: their tail is unknown before the SCF; the "
                    "vacuum was raised from "
                    f"{domain.wall_vacuum / ANGSTROM_TO_BOHR:.1f} to "
                    f"{domain.vacuum / ANGSTROM_TO_BOHR:.1f} ang. This does not "
                    'guarantee Domain_Energy_Tolerance. Read "Density at the '
                    'sphere" after the SCF; with the highest occupied level '
                    "above zero the energy has no limit for a large sphere."
                )
        else:
            vacuum = grid.radius - max(
                float(np.linalg.norm(atom.position)) for atom in placed
            )
            tolerance = auto_boundary_tolerance(
                electrons, domain_energy_tolerance, vacuum
            )
            rule_tolerance_text = f"{tolerance:.1e} Ry"
        if rule_tolerance_text is not None:
            hartree = replace(
                hartree,
                boundary_tolerance=_energy_rydberg(
                    rule_tolerance_text, label="Hartree_Boundary_Tolerance"
                ),
            )
            boundary_tolerance_from = "rule"
        domain_rule_seconds = perf_counter() - rule_started

    # What PARSEC 1.5's pseudo.f90 says, as warnings: a CA file in a PBE run, the minimum number of states
    # Nel/2 + 2 (ierr 213; spinor runs, which PARSEC 1.5 refuses with ierr 158, are not checked).
    warnings.extend(
        _input15_pseudopotential_notes(
            specifications,
            atoms,
            None if (self_consistent_spin_orbit or noncollinear) else states,
            xc_functional,
            potentials=potentials if (default_radius or (default_tolerance and multipole_boundary)) else None,
        )
    )

    if wire_or_slab:
        refused = [
            label
            for condition, label in (
                (
                    any(specification.spin_orbit for specification in specifications.values())
                    or self_consistent_spin_orbit,
                    "spin-orbit coupling",
                ),
                (noncollinear, "non-collinear magnetism"),
                # LDA+U: upot.F90 replicates along the periodic directions
                # (nreplica(1:per)); the combinations the LDA+U code refuses
                # for any periodic system are refused below.
                # usrinputfile.F90:2380-2386: the TS correction needs per = 3.
                (van_der_waals, "Van_Der_Waals"),
                (energy_refinement is not None, "Energy_Refinement"),
                (initial_density_settings.method != "sad", "an initial density other than SAD"),
            )
            if condition
        ]
        if refused:
            raise UnsupportedParsecOptionError(
                "Boundary_Conditions wire or slab is not implemented together with "
                + ", ".join(refused)
            )
    _check_hubbard_and_dispersion(
        specifications,
        van_der_waals=van_der_waals,
        is_periodic=is_periodic,
        spin_polarized=spin_polarized,
        kpoints=monkhorst_pack_dimensions is not None or explicit_kpoint_array is not None,
        spin_orbit=bool(
            self_consistent_spin_orbit
            or noncollinear
            or any(specification.spin_orbit for specification in specifications.values())
        ),
        bands=band_settings is not None,
        energy_refinement=energy_refinement is not None,
        forces=compute_forces or atoms_move,
    )
    # Double_Grid_Order > 1 with SO_PSP and forces: the spin-orbit projector
    # gradients are those of the double-grid projectors (Forces.spin_orbit).
    if any(specification.hubbard_j != 0.0 for specification in specifications.values()):
        warnings.append(
            "ldaplusu_j: the J term uses PARSEC's P3/P4 tensors (upot.F90) on the "
            "occupation matrix of the real d harmonics, as the Fortran does; "
            "its potential matrix is not symmetric there"
        )
    post_processing = _post_processing_input(
        scalar,
        blocks,
        one,
        optional_bool,
        is_periodic=is_periodic,
        wire_or_slab=wire_or_slab,
        periodicity=periodicity,
        periodic_cell=periodic_cell,
        # Fortran converts nscf_kpoint_unit with the input cell (lattvec).
        input_cell=(
            None
            if periodic_cell is None
            else (
                periodic_cell
                if super_cell is None and super_cell_vacuum is None
                else PeriodicCell(lattice_vectors=primitive_vectors)
            )
        ),
        restart_run=restart_run,
        states=states,
        symbols=symbols,
        monkhorst_pack_dimensions=monkhorst_pack_dimensions,
        kpoints=explicit_kpoint_array,
        kpoint_weights=explicit_kpoint_weights,
        spin_orbit=bool(self_consistent_spin_orbit or noncollinear),
        noncollinear=bool(noncollinear),
        warnings=warnings,
        output_gw=output_gw,
        monkhorst_pack_shift=monkhorst_pack_shift,
    )
    if post_processing.nscf is not None and eigensolver.fixed_state_count:
        # calc_nscf.F90:146-147 sets the number of eigenpairs of the NSCF
        # solve itself (nn = nstate): Fix_Number_Eigenvalues changes nothing.
        eigensolver = replace(eigensolver, fixed_state_count=False)
    # Create_DOS with Restart_Run: the SCF runs on the Monkhorst-Pack grid of
    # the DOS (usrinputfile.F90 l.1188-1193) with mpshift = dos_mpshift.  The
    # space group, its wedge and the default of Symmetrize_Charge_Density
    # belong to the shift of the restarted SCF: build them again for it.  A
    # wire or slab finds its group below (_wire_slab_symmetry), with this shift.
    if post_processing.monkhorst_pack_shift is not None and (
        post_processing.monkhorst_pack_shift != monkhorst_pack_shift
    ):
        monkhorst_pack_shift = post_processing.monkhorst_pack_shift
        if not wire_or_slab:
            symmetrize_density, space_group, grid = _periodic_symmetry(
                atoms,
                periodic_cell,
                grid,
                general_periodic_grid=general_periodic_grid,
                ignore_symmetry=ignore_symmetry,
                monkhorst_pack_shift=monkhorst_pack_shift,
                symmetrize_density=optional_bool("symmetrize_charge_density"),
                symmetrize_given="symmetrize_charge_density" in scalar,
                noncollinear=noncollinear,
                warnings=warnings,
                # The restarted SCF runs on the DOS grid (no weighted list).
                search_info=_bulk_symmetry_search_info(
                    periodic_cell,
                    ignore_symmetry=ignore_symmetry,
                    kpoints=True,
                    spinors=bool(
                        noncollinear
                        or self_consistent_spin_orbit
                        or any(
                            specification.spin_orbit for specification in specifications.values()
                        )
                    ),
                    bands=band_settings is not None,
                    nscf=post_processing.nscf is not None,
                    molecular_dynamics=molecular_dynamics.enabled,
                    manual_movement=minimization == "manual",
                    requested_shift=requested_grid_shift,
                ),
            )
            if routines.fortran:
                general_periodic_grid = uses_general_periodic_grid(periodic_cell, grid)
                grid = _fortran_bulk_box_lengths(grid, periodic_cell, general_periodic_grid)
                if general_periodic_grid and double_grid_order > 1:
                    raise UnsupportedParsecOptionError(
                        "Double_Grid_Order > 1 is not implemented on Fortran's general "
                        "periodic grid (non-orthogonal lattice vectors, or steps that "
                        "differ between the lattice directions): its projectors use one grid step"
                    )
            scf = replace(scf, symmetrize_density=bool(symmetrize_density))
            if forces_requested:
                _check_reduced_kpoint_movements(
                    atoms,
                    periodic_cell,
                    space_group,
                    movable=movable,
                    relaxation=minimization in {"bfgs", "simple"},
                    molecular_dynamics=(
                        molecular_dynamics.enabled or minimization == "manual"
                    ),
                    reduced=True,
                    ignore_symmetry=ignore_symmetry,
                )
    monkhorst_pack_dimensions = post_processing.monkhorst_pack_dimensions
    explicit_kpoint_array = post_processing.kpoints
    explicit_kpoint_weights = post_processing.kpoint_weights
    if post_processing.g_space_hartree:
        hartree = replace(hartree, periodic_g_space=True)
    if post_processing.nscf is not None and (
        band_settings is not None
        or compute_forces
        or atoms_move
        or polarizability
    ):
        raise UnsupportedParsecOptionError(
            "nscf_kpoints skips the SCF and everything after it in PARSEC "
            "(parsec.F90 l.1098): bandstruc, forces (Skip_Force: false), "
            "Minimization and Molecular_Dynamics are not run with it; remove them"
        )
    if wire_or_slab:
        symmetrize_density, space_group, grid = _wire_slab_symmetry(
            atoms,
            periodic_cell,
            grid,
            periodicity=periodicity,
            ignore_symmetry=ignore_symmetry,
            monkhorst_pack_shift=monkhorst_pack_shift,
            symmetrize_density=symmetrize_density,
            symmetrize_given="symmetrize_charge_density" in scalar,
            noncollinear=noncollinear,
            # parsec.F90:470-486: what turns the symmetry of a slab on, and
            # what switches every symmetry off.
            spin_orbit=bool(
                noncollinear
                or any(specification.spin_orbit for specification in specifications.values())
            ),
            bands=band_settings is not None,
            nscf=post_processing.nscf is not None,
            no_symmetry=bool(
                molecular_dynamics.enabled or polarizability or minimization == "manual"
            ),
            warnings=warnings,
        )
        if symmetrize_density:
            scf = replace(scf, symmetrize_density=True)
        if compute_forces:
            # The movements a reduced k-point set cannot follow, as for a bulk
            # cell (the wedge forces are completed over the wire/slab group in
            # Forces.wire_slab, which the ionic loop restricts to the
            # operations Move_Flag respects and molecular dynamics drops).
            _check_wire_slab_kpoint_movements(
                atoms,
                periodic_cell,
                grid,
                space_group,
                periodicity=periodicity,
                movable=movable,
                relaxation=minimization in {"bfgs", "simple"},
                molecular_dynamics=molecular_dynamics.enabled or minimization == "manual",
                weighted_kpoints=bool(
                    explicit_kpoint_weights is not None
                    and not np.allclose(explicit_kpoint_weights, explicit_kpoint_weights[0])
                ),
                monkhorst_pack_shift=monkhorst_pack_shift,
                ignore_symmetry=ignore_symmetry,
                slab_symmetry_on=bool(
                    band_settings is not None
                    or post_processing.nscf is not None
                    or noncollinear
                    or any(
                        specification.spin_orbit for specification in specifications.values()
                    )
                ),
            )
    has_kpoints = monkhorst_pack_dimensions is not None or explicit_kpoint_array is not None
    has_soc_species = any(
        specification.spin_orbit for specification in specifications.values()
    )
    # LDA+U of a wire or slab with k-points, spin polarization or a band
    # structure: the Bloch projectors replicate along the periodic directions
    # alone (nreplica(1:per), Hamiltonian.hubbard.hubbard_replica_lattice) and
    # the occupations of an irreducible wedge are averaged over the wire/slab
    # group of alatt (hubbard_symmetry_lattice).
    # A wire or slab relaxes and moves with spin polarization and k-points as
    # a bulk cell does: the spin k-point loop starts every structure from the
    # channel potentials of the one before (Relaxation.wire_slab_backend).
    # The drivers of a plain SCF loop (run_scf, and run_spin_scf for an
    # isolated spin-polarized input without spin-orbit species): the others
    # (k-points, spin-orbit, non-collinear, periodic spin) keep no fields of
    # their iterations and take no starting potential, but for the scalar
    # k-point loop.
    plain_loop = not (
        has_kpoints
        or self_consistent_spin_orbit
        or noncollinear
        or (spin_polarized and (is_periodic or has_soc_species))
    )
    gw = None
    if output_gw:
        gw = _gw_settings(
            post_processing,
            atoms=atoms,
            periodic_cell=periodic_cell,
            is_periodic=is_periodic,
            wire_or_slab=wire_or_slab,
            spinors=bool(self_consistent_spin_orbit or noncollinear or has_soc_species),
            bands=band_settings is not None,
            moves=atoms_move,
            forces=compute_forces,
            monkhorst_pack_dimensions=monkhorst_pack_dimensions,
            monkhorst_pack_shift=monkhorst_pack_shift,
            explicit_kpoints=explicit_kpoint_array is not None,
            space_group=space_group,
            ignore_symmetry=ignore_symmetry,
            warnings=warnings,
        )
    output_files = _output_file_settings(scalar, blocks, one, optional_bool, has_kpoints)
    # Save_Intermediate_Charge_Density / Save_Intermediate_Eigenvalues: every
    # SCF driver hands its iterations to the output files (parsec.F90
    # l.1381-1402 writes them in PARSEC's one SCF loop for every kind of run).
    # A wire or slab writes and reads parsec.dat as wfnsave.F90 /
    # restart_run.F90 do for pbc%per = 1, 2 (stype(3) = 2, 3; Output/parsec_dat.py).
    # Fortran's general periodic grid (non-orthogonal cells) has its parsec.dat
    # (shift and per-direction steps, Output/parsec_dat.py), so Restart_Run
    # and Save_Intermediate_Charge_Density work there as on the other grids.
    if restart_from_wfndat and not restart_run:
        warnings.append(
            "Restart_From_Wfndat is read only together with Restart_Run: true "
            "(parsec.F90); this run starts from the atomic potentials"
        )
        restart_from_wfndat = False
    if restart_from_wfndat and (
        has_kpoints
        or post_processing.nscf is not None
        or post_processing.dos is not None
    ):
        # restart.F90 has no k-points (kpnum = 1).
        raise UnsupportedParsecOptionError(
            "Restart_From_Wfndat: wfn.dat has no k-points (restart.F90); it "
            "restarts a cluster or a Gamma-point cell, without nscf_kpoints or "
            "Create_DOS"
        )
    if restart_from_wfndat and (self_consistent_spin_orbit or noncollinear):
        # The spin-orbit loops restart from the two spin components of a
        # spin-orbit parsec.dat; wfn.dat holds one potential per channel.
        raise UnsupportedParsecOptionError(
            "Restart_From_Wfndat with spin-orbit coupling or non-collinear "
            "magnetism is not implemented: wfn.dat holds no spinor components; "
            "Restart_Run restarts a spin-orbit run from its parsec.dat"
        )
    if restart_run:
        # A non-self-consistent run reads the potential of every spin channel
        # and runs no SCF loop, so it needs none of the restartable drivers.
        # A self-consistent spin-orbit run (isolated, or a bulk cell at Gamma
        # or with k-points) restarts from the parsec.dat of a spin-orbit run
        # (restart_run.F90 with stype(2) = 2).
        isolated_spin_orbit = bool(
            self_consistent_spin_orbit and not noncollinear and not wire_or_slab
        )
        if post_processing.nscf is None and not (plain_loop or isolated_spin_orbit or (has_kpoints and not spin_polarized and not self_consistent_spin_orbit and not noncollinear)):
            raise UnsupportedParsecOptionError(
                "Restart_Run is implemented for the SCF loop of a cluster or a "
                "periodic cell (Gamma point or scalar k-points), for an isolated "
                "spin-polarized input and for a self-consistent spin-orbit run; "
                "not with non-collinear magnetism or periodic spin polarization"
            )
        if initial_density_settings.method != "sad":
            raise UnsupportedParsecOptionError(
                "Restart_Run takes the potential of parsec.dat; Initial_Density: "
                f"{initial_density_settings.method} would not be used: remove it"
            )
        if mlff is not None:
            raise UnsupportedParsecOptionError(
                "Restart_Run takes the potential of parsec.dat for the structure of "
                "the input; Initial_Structure: mlff would move the atoms first"
            )
        if not is_periodic and minimization != "manual":
            from ..V_ion import center_cluster_geometry

            # initial.f90:133-134 centres a cluster only with istart = 0, and
            # parsec.dat stores no coordinates (wfnsave.F90): PARSEC restarts
            # the stored potential at the input coordinates, and so does this
            # port (the same fixed points as the Fortran binary, also when the
            # first run had moved the atoms).
            placed = center_cluster_geometry(atoms)
            moved = max(
                float(np.linalg.norm(np.asarray(a.position) - np.asarray(b.position)))
                for a, b in zip(atoms, placed)
            )
            if moved > 0.0:
                warnings.append(
                    "Restart_Run: the atom coordinates are used as the input gives "
                    "them (PARSEC centres a cluster only in a first run); a first "
                    f"run would have moved them by {moved:.4f} bohr, so the "
                    "parsec.dat of such a run belongs to other coordinates: give "
                    "the centred coordinates that run printed (PARSEC: 'The new "
                    "atom coordinates are'; this port: 'initial coordinates') to "
                    "restart from its potential"
                )

    grid, scf = _periodic_potential_conventions(
        scalar,
        one,
        optional_bool,
        grid,
        scf,
        is_periodic=is_periodic,
        wire_or_slab=wire_or_slab,
        bulk=periodicity == 3,
        warnings=warnings,
    )
    if is_periodic and not wire_or_slab and not routines.fortran:
        refusal = _default_routines_periodic_refusal(
            periodic_cell,
            grid,
            atoms,
            slab=andrew_slab,
            kpoints=explicit_kpoint_array,
            kpoint_weights=explicit_kpoint_weights,
            monkhorst_pack_dimensions=monkhorst_pack_dimensions,
            spin_polarized=spin_polarized,
            ionic=bool(compute_forces or atoms_move),
            functional=scf.xc_functional,
        )
        if refusal is not None:
            fall_back(refusal)
    if routines.warning() is not None:
        warnings.append(routines.warning())
    if andrew_slab:
        if (
            self_consistent_spin_orbit
            or noncollinear
            or any(specification.spin_orbit for specification in specifications.values())
        ):
            # Andrew Grieder's slab refuses them
            fall_back("spin-orbit coupling or non-collinear magnetism of a slab")
        # in-plane positions are folded into the cell (they are periodic)
        inverse_plane = np.linalg.inv(periodic_cell.lattice_vectors[:2, :2])
        folded = []
        for atom in atoms:
            position = np.array(atom.position, dtype=float)
            fraction = position[:2] @ inverse_plane
            fraction -= np.floor(fraction + 1.0e-12)
            position[:2] = fraction @ periodic_cell.lattice_vectors[:2, :2]
            folded.append(replace(atom, position=position))
        atoms = folded
        if recenter and atoms:
            # Fortran (initial.f90) centres a slab along z when the atoms are off centre by more than 0.1 bohr
            z_values = [float(atom.position[2]) for atom in atoms]
            shift = 0.5 * (min(z_values) + max(z_values))
            if abs(shift) > 0.1:
                warnings.append(
                    f"atom coordinates were shifted by {-shift:.6f} bohr along z so that the slab is centred at the origin"
                )
                atoms = [
                    replace(atom, position=np.asarray(atom.position, dtype=float) - np.array([0.0, 0.0, shift]))
                    for atom in atoms
                ]
        height = float(periodic_cell.lattice_vectors[2, 2])
        if any(abs(float(atom.position[2])) >= 0.5 * height for atom in atoms):
            # Andrew Grieder's slab preparation takes the atoms inside its box only (|z| < R)
            fall_back("a slab atom at |z| >= Boundary_Sphere_Radius")

    # FDET of PARSEC 1.5 (Potential_Field) and pot.dat (Write_Potential_Field).
    fdet = _fdet_settings(
        scalar,
        one,
        optional_bool,
        source,
        warnings,
        is_periodic=is_periodic,
        scf=scf,
        point_charges=point_charges,
        radius_given="boundary_sphere_radius" in scalar,
        polarizability=polarizability,
        energy_refinement=energy_refinement,
        nscf=post_processing.nscf,
        specifications=specifications,
        ignore_symmetry=ignore_symmetry,
        minimization=minimization,
        molecular_dynamics=molecular_dynamics.enabled,
        extrapolation=potential_extrapolation,
        grid_shift=tuple(float(value) for value in getattr(grid, "shift", (0.0, 0.0, 0.0))),
    )
    mixing = _check_mixing_preconditioner(
        mixing,
        scf,
        periodicity=periodicity,
        specifications=specifications,
        embedding=fdet.embedding is not None,
        requested=_normalize_label(one("mixing_preconditioner", "auto")) != "auto",
    )
    if scf.is_libxc:
        _check_libxc_request(scf, specifications=specifications, embedding=fdet.embedding is not None)
    if scf.is_meta_gga:
        _check_meta_gga_request(
            scf,
            is_periodic=is_periodic,
            specifications=specifications,
            energy_refinement=energy_refinement,
            embedding=fdet.embedding is not None,
            polarizability=polarizability,
            nscf=post_processing.nscf is not None,
            output_gw=output_gw,
            bands=band_settings is not None,
            restart=bool(restart_run),
        )
    if scf.is_hybrid:
        _check_hybrid_request(
            scf,
            is_periodic=is_periodic,
            periodicity=periodicity,
            specifications=specifications,
            energy_refinement=energy_refinement,
            embedding=fdet.embedding is not None,
            polarizability=polarizability,
            nscf=post_processing.nscf is not None,
            output_gw=output_gw,
            bands=band_settings is not None,
        )
    afm_scan = _afm_scan_settings(
        scalar,
        blocks,
        one,
        optional_bool,
        embedding=fdet.embedding,
        coordinate_factor=coordinate_factor,
        spacing=float(getattr(grid, "spacing", 0.0) or 0.0),
        minimization=minimization,
        molecular_dynamics=molecular_dynamics.enabled,
        atoms=atoms,
    )
    if afm_scan is not None:
        # The scan is a manual sequence of rigid tip displacements (Embedding.afm),
        # whose first structure is the first scan point.
        structures = afm_scan.structures([atom.position for atom in atoms])
        _check_scan_structures(afm_scan, structures, atoms, grid, tip_sphere=fdet.embedding.tip_sphere)
        atoms = [replace(atom, position=structures[0][index]) for index, atom in enumerate(atoms)]
        relaxation = RelaxationSettings(method="manual", max_movements=len(structures) - 1)
        movable = [True] * len(atoms)
        compute_forces = True
    if fdet.embedding is not None:
        # The frozen field stays where the sample run left it: the tip is
        # never recentred (initial.f90 of PARSEC 1.5 recentres it without
        # the field), and the field breaks the point group of the tip atoms.
        if recenter and not is_periodic:
            warnings.append(
                "FDET (Potential_Field): the tip atoms are not recentred, so that they stay in the frame of "
                "the frozen sample field"
            )
        recenter = False
        # Recenter_Atoms_In_Aperiodic_Directions (PARSEC 1.5) as .false.: also a
        # molecular-dynamics restart keeps the frame of the field.
        recenter_aperiodic = False
        use_symmetry = False

    problem = SinglePointInput(
        atoms=atoms,
        pseudopotentials=specifications,
        grid=grid,
        periodic_cell=periodic_cell,
        scf=scf,
        hartree=hartree,
        eigensolver=eigensolver,
        mixing=mixing,
        initial_density_settings=initial_density_settings,
        recenter_geometry=recenter and not is_periodic,
        monkhorst_pack_dimensions=monkhorst_pack_dimensions,
        kpoints=explicit_kpoint_array,
        kpoint_weights=explicit_kpoint_weights,
        bands=band_settings,
        point_charges=point_charges,
        **({"charged_sheets": charged_sheets} if charged_sheets else {}),
        monkhorst_pack_shift=monkhorst_pack_shift,
        space_group=space_group,
        **({} if fdet.embedding is None else {"embedding": fdet.embedding}),
    )
    output_all_states = optional_bool("output_all_states")
    output_level = _integer(one("output_level", "1"), label="Output_Level")
    # --esdf-lenient: the duplicates found since the warnings were collected
    warnings.extend(lenient_notes[lenient_reported:])
    translation = ParsecInputTranslation(
        source=source,
        problem=problem,
        warnings=tuple(warnings),
        output_all_states=output_all_states,
        output_level=output_level,
        ignore_symmetry=ignore_symmetry,
        domain_energy_tolerance=domain_energy_tolerance,
        domain=domain,
        boundary_tolerance_from=boundary_tolerance_from,
        domain_rule_seconds=domain_rule_seconds,
        relaxation=relaxation,
        movable=tuple(movable),
        compute_forces=compute_forces,
        use_symmetry=use_symmetry,
        molecular_dynamics=molecular_dynamics if molecular_dynamics.enabled else None,
        potential_extrapolation=potential_extrapolation,
        md_output_interval=md_output_interval,
        mlff=mlff,
        energy_refinement=energy_refinement,
        polarizability_field=electric_field if polarizability else None,
        point_charge_type_counts=point_charge_type_counts,
        magnetic_field_gauss=magnetic_field / 2.5e9,
        complex_wavefunctions=complex_wavefunctions,
        output_files=output_files,
        restart_run=restart_run,
        restart_from_wfndat=restart_from_wfndat,
        fix_number_eigenvalues=fix_number_eigenvalues,
        dynamic_tolerance_off_for_restart=dynamic_tolerance_off_for_restart,
        super_cell=super_cell,
        super_cell_vacuum=super_cell_vacuum,
        nscf=post_processing.nscf,
        dos=post_processing.dos,
        gw=gw,
        slab_shift=slab_shift,
        routines=routines,
        write_potential_field=fdet.write_potential_field,
        afm_scan=afm_scan,
        grid_shift=requested_grid_shift,
        atoms_outside_cell=atoms_outside_cell,
        recenter_aperiodic=recenter_aperiodic,
    )
    if input15.complete_input:
        # PARSEC 1.5's Generate_Complete_Parsec_In: the effective input (bak_complete_parsec.in)
        from .complete_input import complete_input_text

        translation = replace(
            translation,
            complete_input=complete_input_text(
                translation,
                scalar,
                blocks,
                # the cell of the lattice-unit k-point lists of an input with a Super_Cell (Kpoint_List, nscf_kpoints)
                kpoint_cell=(
                    PeriodicCell(lattice_vectors=primitive_vectors)
                    if primitive_vectors is not None and (super_cell is not None or super_cell_vacuum is not None)
                    else None
                ),
            ),
        )
    return translation


def _periodic_symmetry(
    atoms,
    periodic_cell: PeriodicCell,
    grid: PeriodicGridSettings,
    *,
    general_periodic_grid: bool,
    ignore_symmetry: bool,
    monkhorst_pack_shift,
    symmetrize_density: bool,
    symmetrize_given: bool,
    noncollinear: bool,
    warnings: list[str],
    forces_of_weighted_kpoints: bool = False,
    search_info: int = 0,
):
    """The space group of a periodic input and what it decides.

    Returns ``(symmetrize_density, space_group, grid)``:

    * ``Symmetrize_Charge_Density`` defaults to true where Fortran's k-point
      path is taken (an input with ``Monkhorst_Pack_Shift``), as
      ``usrinputfile.F90:2548-2552`` sets it for every k-point run; other
      periodic inputs keep this port's default, false;
    * the space group (``symmetries.f90``; the identity with
      ``Ignore_Symmetry``) is kept in the problem where the irreducible wedge,
      the symmetrization or the forces use it (``symm_forces.f90``: the forces
      of an unequally weighted ``Kpoint_List`` are completed over it,
      ``forces_of_weighted_kpoints``), ``None`` otherwise;
    * ``search_info`` is the ``ierr`` that ``parsec.F90:464-492`` hands to
      ``symmetries.f90`` (:func:`_bulk_symmetry_search_info`).  2 (molecular
      dynamics, ``manual`` movements): no search, the group is the identity
      (``skipping non-trivial symmetry operations``), so ``adjust_step`` sees
      no translation and with ``Monkhorst_Pack_Shift`` the wedge is that of
      time reversal alone.  0 (a Gamma-point run on a grid shifted by 0.5,
      i.e. orthogonal lattice vectors, without spinors, bands or NSCF):
      ``symgen.f90:209-210`` checks with ``symchk`` that the operations form
      a group modulo the lattice and keeps the identity alone when they do
      not (``ERROR: repeated symmetry operations``; a centred structure in a
      conventional cell, whose translations are fixed only up to its
      centring vectors: the 8-atom diamond cube, also stretched along z).
      The problem then carries no group, as for the identity.  1 (k-points,
      oblique cells, spinors, bands, NSCF): the whole group, unchecked;
    * the grids of the Fortran-derived routines (the orthorhombic one and
      Fortran's general grid) get the fractional translations ``adjust_step``
      makes their steps commensurate with (``inipbc.f90:327-429``): the group
      is searched for them whenever ``symmetries.f90`` searches (not with
      ``Ignore_Symmetry`` or ``search_info`` 2), also where the problem does
      not keep it.  A symmetrization is refused by name when no spacing is
      commensurate with a translation
      (:func:`parsec_python.Grid.pbc.adjust_step`).  Andrew Grieder's grid
      takes no translations (``general_periodic_grid`` is the provisional
      choice of the caller and decides nothing here since PARSEC 1.5's parity
      round; the caller decides the grid again from the translations).

    A non-collinear calculation is refused with a non-trivial group: its
    magnetization is a vector field, which the scalar average of
    ``symm_scalar`` does not treat.
    """

    from ..Grid.space_group import (
        find_space_group,
        grid_commensurability_problem,
        identity_space_group,
        skipped_space_group,
    )
    from ..Grid.symmetry_analysis import symmetry_check

    fortran_kpoints = monkhorst_pack_shift is not None
    if not symmetrize_given and fortran_kpoints:
        symmetrize_density = True
    # Every grid of the Fortran-derived routines takes Fortran's counts with the translations of the group.
    fortran_grid = bool(getattr(grid, "fortran_routines", False))
    searched_for_grid = fortran_grid and not ignore_symmetry and search_info != 2
    if not (
        symmetrize_density
        or fortran_kpoints
        or general_periodic_grid
        or forces_of_weighted_kpoints
        or searched_for_grid
    ):
        return symmetrize_density, None, grid
    failed_check = False
    if ignore_symmetry:
        group = identity_space_group()
    elif search_info == 2:
        # parsec.F90 l.483-485: no symmetry operations for molecular dynamics
        # and manual movements (ierr = 2): adjust_step sees no translation
        # and the wedge of a k-point grid is that of time reversal alone.
        group = skipped_space_group()
        if forces_of_weighted_kpoints:
            # The structure's own group instead, so that the movement check
            # (_check_reduced_kpoint_movements) refuses by name the unequally
            # weighted Kpoint_List (a wedge of that group) such a run cannot
            # follow.
            try:
                group = find_space_group(periodic_cell.lattice_vectors, atoms)
            except ValueError as error:
                raise ParsecInputError(f"symmetry operations: {error}") from error
    else:
        try:
            group = find_space_group(periodic_cell.lattice_vectors, atoms)
        except ValueError as error:
            raise ParsecInputError(f"symmetry operations: {error}") from error
        if search_info == 0 and symmetry_check(group).error:
            # symgen.f90 l.209-210: the operations do not form a group modulo
            # the lattice (symchk), and PARSEC keeps the identity alone.
            group = identity_space_group(ignored=False)
            failed_check = True
        if fortran_grid:
            grid = replace(grid, fractional_translations=group.commensurate_translations())
    if symmetrize_density and noncollinear:
        if group.order > 1:
            # PARSEC (newrho.F90, parsec.F90:1223-1232) averages rho and the
            # local-frame densities (rho +- |m|)/2 as scalars and keeps the
            # direction of m: SCF/noncollinear.py does the same.
            warnings.append(
                "Symmetrize_Charge_Density with Non_Collinear_magnetism averages "
                "the density and |m| as scalars and keeps the direction of the "
                "magnetization, as PARSEC does"
            )
        else:
            if symmetrize_given:
                warnings.append(
                    "Symmetrize_Charge_Density changes nothing here: the group has "
                    "the identity alone"
                )
            symmetrize_density = False
    problem = None
    if (general_periodic_grid or fortran_grid) and group.order > 1:
        from ..Grid.pbc import fortran_grid_counts

        try:
            counts = [int(value) for value in fortran_grid_counts(periodic_cell, grid)[0]]
        except ValueError:
            counts = None  # the grid builder says why
        if counts is not None:
            problem = grid_commensurability_problem(group, counts, grid.shift)
        if problem is not None and not symmetrize_density:
            warnings.append(
                "symmetry operations: " + problem + ": no grid spacing near "
                "Grid_Spacing is commensurate with that fractional translation; the "
                f"grid has {'x'.join(str(count) for count in counts)} points, PARSEC 1.5's "
                "(nint(L/h) points with the step L/n along such a direction, and the "
                "other directions commensurate down to the smallest of these steps)"
            )
    if symmetrize_density and (general_periodic_grid or fortran_grid):
        if problem is not None:
            raise UnsupportedParsecOptionError(
                "Symmetrize_Charge_Density: " + problem + ": no grid spacing is "
                "commensurate with that fractional translation (PARSEC's adjust_step "
                "then keeps the input spacing, and symm_scalar averages over points "
                "that are not images of each other).  Set Ignore_Symmetry: true, or "
                "Symmetrize_Charge_Density: false (with Monkhorst_Pack_Shift the "
                "density of the irreducible wedge then misses its rotated k-points)"
            )
    if symmetrize_density and not (general_periodic_grid or fortran_grid):
        # (Andrew Grieder's grid of a DOS restart reaches this check with the rule of the earlier orthorhombic grid
        # of the Fortran-derived routines.)
        side_lengths = np.diag(periodic_cell.lattice_vectors)
        counts = np.round(side_lengths / grid.spacing).astype(np.int64)
        problem = grid_commensurability_problem(group, counts, (0.0, 0.0, 0.0))
        if problem is not None:
            raise UnsupportedParsecOptionError(
                "Symmetrize_Charge_Density: " + problem + ".  PARSEC chooses its "
                "grid spacing commensurate with the fractional translations "
                "(adjust_step); this orthorhombic cell keeps the grid of this "
                "port (n = round(L/h), no shift).  Choose a Grid_Spacing for which "
                "every n_i times the translations is a whole number, or set "
                "Symmetrize_Charge_Density: false (with Monkhorst_Pack_Shift the "
                "density of the irreducible wedge then misses its rotated "
                "k-points) or Ignore_Symmetry: true"
            )
    if fortran_kpoints and not symmetrize_density and group.order > 1:
        warnings.append(
            "Symmetrize_Charge_Density: false with a Monkhorst_Pack_Shift grid: "
            "the irreducible k-points alone do not give the density of the "
            "cell, which only the average over the space group restores "
            "(PARSEC does the same)"
        )
    if forces_of_weighted_kpoints and not symmetrize_density and group.order > 1:
        warnings.append(
            "Kpoint_List with unequal weights and forces: the forces of the listed "
            "k-points are completed over the space group (an irreducible wedge), "
            "but without Symmetrize_Charge_Density: true the density is that of "
            "the listed points alone"
        )
    keep = (
        symmetrize_density or fortran_kpoints or forces_of_weighted_kpoints
    ) and not failed_check
    return symmetrize_density, (group if keep else None), grid


def _default_routines_wedge(
    atoms,
    periodic_cell: PeriodicCell,
    grid: PeriodicGridSettings,
    mesh,
    *,
    ignore_symmetry: bool,
    symmetrize_density: bool,
    symmetrize_given: bool,
    warnings: list[str],
):
    """The symmetry of a ``Monkhorst_Pack_Shift`` mesh of a bulk cell on the default routines.

    ``mesh`` is ``(Monkhorst_Pack_Grid, Monkhorst_Pack_Shift)``.  Returns ``(symmetrize_density, space_group)``; the
    caller keeps the mesh as Fortran's grid ``k = (i + s)/n`` (``monkhorst_pack_shift``), which
    :func:`parsec_python.Grid.kpoints.kpoint_sampling_details` reduces by ``space_group`` (the identity for
    ``None``) and time reversal, as ``irrbz`` does on the Fortran-derived routines.  Their grid follows the space
    group (``adjust_step``); Andrew Grieder's keeps the points of ``Grid_Spacing``, so the group here is the subgroup
    of the crystal's operations that map his grid onto itself
    (:func:`parsec_python.Grid.space_group.find_grid_space_group`) and the k-point mesh onto itself
    (:func:`parsec_python.Grid.kpoints.mesh_preserving_operations`: then the wedge with the averaged density is the
    full mesh, which these routines took before), with which the density is averaged on that grid
    (``Symmetrize_Charge_Density``, true by default as on Fortran's k-point path; ``usrinputfile.F90:2548-2552``).
    Diamond in the cube with 18 points per edge keeps the 24 operations of ``Td`` about an atom: with time reversal
    the wedge of ``Oh`` (Gamma-centred 2^3, 3^3, 4^3 meshes: 4, 4 and 10 points).

    ``Ignore_Symmetry`` or ``Symmetrize_Charge_Density: .false.``: time reversal alone (``k`` and ``-k`` have the
    same density, which needs no average).
    """

    from ..Grid.kpoints import mesh_preserving_operations
    from ..Grid.pbc import default_periodic_grid_counts
    from ..Grid.space_group import find_grid_space_group, find_space_group

    if ignore_symmetry or (symmetrize_given and not symmetrize_density):
        if ignore_symmetry and symmetrize_given and symmetrize_density:
            warnings.append(
                "Symmetrize_Charge_Density changes nothing with Ignore_Symmetry: the Monkhorst_Pack_Shift mesh is "
                "reduced by time reversal alone"
            )
        return symmetrize_density, None
    lattice_vectors = np.asarray(periodic_cell.lattice_vectors, dtype=float)
    counts = default_periodic_grid_counts(periodic_cell, grid)
    try:
        on_grid = find_grid_space_group(lattice_vectors, atoms, counts, grid.shift)
        crystal = find_space_group(lattice_vectors, atoms)
    except ValueError as error:
        raise ParsecInputError(f"symmetry operations: {error}") from error
    group = mesh_preserving_operations(on_grid, *mesh)
    if group.order <= 1:
        return symmetrize_density, None
    if crystal.order > group.order:
        dropped = []
        if on_grid.order < crystal.order:
            dropped.append(
                f"{crystal.order - on_grid.order} have fractional translations of no whole number of grid steps "
                "(the grid of Grid_Spacing, " + "x".join(str(int(value)) for value in counts) + " points)"
            )
        if group.order < on_grid.order:
            dropped.append(f"{on_grid.order - group.order} more do not map the k-point mesh onto itself")
        warnings.append(
            f"Monkhorst_Pack_Shift: the k-point mesh is reduced to the irreducible wedge of {group.order} of the "
            f"{crystal.order} operations of the space group and time reversal, and the density is symmetrized "
            "with them (" + "; ".join(dropped) + ")"
        )
    return True, group


def _default_routines_list_symmetry(
    atoms,
    periodic_cell: PeriodicCell,
    grid: PeriodicGridSettings,
    *,
    ignore_symmetry: bool,
    symmetrize_density: bool,
    warnings: list[str],
):
    """The density average of a ``Kpoint_List`` of unequal weights of a bulk single point on the default routines.

    Returns ``(symmetrize_density, space_group)``.  The k-point drivers weight every level by its point (the Fermi
    level, the density and the band energy, ``flevel.f90``/``newrho.F90``/``totnrg.f90``), so the list runs as
    given, as in PARSEC, whose ``Symmetrize_Charge_Density`` is false by default for such a list: the density is that
    of the listed points, which is the density of the cell where the list samples the whole zone.  A list that is an
    irreducible wedge needs the average over the operations of its reduction: ``Symmetrize_Charge_Density: .true.``
    takes those of the crystal that map the grid of ``Grid_Spacing`` onto itself
    (:func:`parsec_python.Grid.space_group.find_grid_space_group`; the weights must then be those of that group's
    stars, which the input vouches for).  Without it, or with ``Ignore_Symmetry``, nothing is averaged, and a warning
    says so.
    """

    from ..Grid.pbc import default_periodic_grid_counts
    from ..Grid.space_group import find_grid_space_group

    if not symmetrize_density or ignore_symmetry:
        warnings.append(
            "Kpoint_List with unequal weights: the density is that of the listed points with their weights; for "
            "an irreducible wedge give Symmetrize_Charge_Density: .true. (without Ignore_Symmetry), which averages "
            "it over the operations of the space group that map the grid onto itself"
        )
        return symmetrize_density, None
    lattice_vectors = np.asarray(periodic_cell.lattice_vectors, dtype=float)
    counts = default_periodic_grid_counts(periodic_cell, grid)
    try:
        group = find_grid_space_group(lattice_vectors, atoms, counts, grid.shift)
    except ValueError as error:
        raise ParsecInputError(f"symmetry operations: {error}") from error
    if group.order <= 1:
        warnings.append(
            "Symmetrize_Charge_Density changes nothing here: no operation of the space group but the identity maps "
            "the grid of Grid_Spacing onto itself"
        )
        return symmetrize_density, None
    return True, group


def _bulk_symmetry_search_info(
    periodic_cell: PeriodicCell,
    *,
    ignore_symmetry: bool,
    kpoints: bool,
    spinors: bool,
    bands: bool,
    nscf: bool,
    molecular_dynamics: bool,
    manual_movement: bool,
    requested_shift=None,
) -> int:
    """The ``ierr`` ``parsec.F90:464-492`` hands to ``symmetries.f90`` for a bulk cell.

    :func:`parsec_python.Grid.symmetry_analysis.symmetry_search_mode` with
    Fortran's grid shift (``usrinputfile.F90:786-794``: 0.5 along orthogonal
    lattice vectors, 0 along oblique ones and with ``Ignore_Symmetry``, or the
    ``Grid_Shift`` block ``requested_shift`` of PARSEC 1.5), which the report of
    the run uses as well.
    """

    from ..Grid.pbc import periodic_grid_shift
    from ..Grid.symmetry_analysis import symmetry_search_mode

    return symmetry_search_mode(
        ignore_symmetry=ignore_symmetry,
        kpoints=kpoints,
        grid_shift=periodic_grid_shift(
            periodic_cell.lattice_vectors,
            ignore_symmetry=ignore_symmetry,
            requested=requested_shift,
        ),
        periodicity=3,
        spinors=spinors,
        bands=bands,
        nscf=nscf,
        molecular_dynamics=molecular_dynamics,
        polarizability=False,
        manual_movement=manual_movement,
        non_spherical_cluster=False,
    ).info


def _wire_slab_symmetry(
    atoms,
    periodic_cell: PeriodicCell,
    grid: PeriodicGridSettings,
    *,
    periodicity: int,
    ignore_symmetry: bool,
    monkhorst_pack_shift,
    symmetrize_density: bool,
    symmetrize_given: bool,
    noncollinear: bool,
    spin_orbit: bool,
    bands: bool,
    nscf: bool,
    no_symmetry: bool,
    warnings: list[str],
):
    """The space group of a wire or slab and what it decides.

    Returns ``(symmetrize_density, space_group, grid)``.  Fortran's choice
    (``parsec.F90:470-486``): ``Ignore_Symmetry`` suppresses the search; a
    slab gets none (``ierr = 2``) unless band structure, spin-orbit /
    non-collinear or an NSCF run sets ``ierr = 1`` after it; molecular
    dynamics, Polarizability and manual movements get none.  Otherwise
    ``symmetries.f90`` searches with ``alatt`` = ``latt_vec`` and ``4 rmax``
    along the confined directions
    (:func:`parsec_python.Grid.space_group.find_wire_slab_space_group`); the
    operations that do not map the confined grid onto itself are dropped (a
    Fortran defect: with ``bandstruc`` a slab's symmetrization averages
    over points that are not images of each other).  The group is always
    kept in the problem (its block is printed, the forces are averaged over
    it); the fractional translations along the periodic directions make
    ``adjust_step`` commensurate, ``Monkhorst_Pack_Shift`` reduces the
    k-points with it and turns ``Symmetrize_Charge_Density`` on by default
    (``usrinputfile.F90:2548-2552``).
    """

    from ..Grid.space_group import (
        find_wire_slab_space_group,
        identity_space_group,
        skipped_space_group,
    )

    fortran_kpoints = monkhorst_pack_shift is not None
    if not symmetrize_given and fortran_kpoints:
        symmetrize_density = True
    if ignore_symmetry:
        group = identity_space_group()
    elif no_symmetry or (periodicity == 2 and not (bands or spin_orbit or nscf)):
        group = skipped_space_group()
    else:
        try:
            group, dropped = find_wire_slab_space_group(
                periodic_cell.lattice_vectors, periodicity, grid.confinement_radius, atoms
            )
        except ValueError as error:
            raise ParsecInputError(f"symmetry operations: {error}") from error
        if dropped:
            warnings.append(
                f"{dropped} symmetry operation(s) of PARSEC's search are not used: "
                "they translate along a confined direction (or mix it with a "
                "periodic one) and do not map the wire or slab grid onto itself "
                "(PARSEC applies them anyway)"
            )
        translations = group.commensurate_translations(periodicity)
        if any(translations):
            grid = replace(
                grid,
                fractional_translations=tuple(translations) + ((),) * (3 - periodicity),
            )
    if symmetrize_density and noncollinear and group.order > 1:
        raise UnsupportedParsecOptionError(
            "Symmetrize_Charge_Density with Non_Collinear_magnetism is not "
            "implemented: the magnetization is a vector field, which the "
            "scalar average of symm_scalar does not treat (PARSEC averages "
            "its components as scalars).  Set Symmetrize_Charge_Density: "
            "false or Ignore_Symmetry: true"
        )
    if fortran_kpoints and not symmetrize_density and group.order > 1:
        warnings.append(
            "Symmetrize_Charge_Density: false with a Monkhorst_Pack_Shift grid: "
            "the irreducible k-points alone do not give the density of the "
            "cell, which only the average over the space group restores "
            "(PARSEC does the same)"
        )
    return symmetrize_density, group, grid


def _check_reduced_kpoint_movements(
    atoms,
    periodic_cell: PeriodicCell,
    space_group,
    *,
    movable,
    relaxation: bool,
    molecular_dynamics: bool,
    reduced: bool,
    ignore_symmetry: bool,
    symmetry_lattice=None,
) -> None:
    """Refuse movements a reduced k-point set cannot follow.

    The forces of a k-point wedge are completed over the space group
    (:func:`parsec_python.Forces.periodic.reduced_kpoint_operations`), which
    holds only while the structure keeps its space group.  A relaxation keeps
    it when its movable flags are mapped onto movable flags by every
    operation; molecular dynamics and ``manual`` movements keep no group
    (their ``Monkhorst_Pack_Shift`` wedge is the identity's, see
    :func:`_periodic_symmetry`), which an unequally weighted ``Kpoint_List``
    would need.  (A symmetrized density with an unreduced k-point set is
    left as it was.)  ``symmetry_lattice`` is the lattice of the group's
    ``M`` and ``t`` where it is not the cell's (``alatt`` of a wire or slab,
    :func:`_check_wire_slab_kpoint_movements`).
    """

    if space_group is None or space_group.order <= 1 or ignore_symmetry or not reduced:
        return
    if molecular_dynamics:
        raise UnsupportedParsecOptionError(
            "Molecular_Dynamics and manual movements use no symmetry (as PARSEC), "
            "so the structure leaves the space group that an unequally weighted "
            "Kpoint_List (an irreducible wedge) relies on; give a complete "
            "k-point set with equal weights, or Monkhorst_Pack_Shift (its wedge "
            "is then that of time reversal alone)"
        )
    if not relaxation:
        return
    from ..Grid.space_group import space_group_atom_images

    mask = np.asarray(movable, dtype=bool)
    images = space_group_atom_images(
        space_group,
        periodic_cell.lattice_vectors if symmetry_lattice is None else symmetry_lattice,
        atoms,
    )
    if any(not np.array_equal(mask[row], mask) for row in images):
        raise UnsupportedParsecOptionError(
            "Move_Flag fixes atoms that a space-group operation exchanges with "
            "movable ones: the relaxation would leave the space group that the "
            "irreducible k-point wedge relies on; "
            "move them all, or set Ignore_Symmetry: true"
        )


def _check_wire_slab_kpoint_movements(
    atoms,
    periodic_cell: PeriodicCell,
    grid,
    space_group,
    *,
    periodicity: int,
    movable,
    relaxation: bool,
    molecular_dynamics: bool,
    weighted_kpoints: bool,
    monkhorst_pack_shift,
    ignore_symmetry: bool,
    slab_symmetry_on: bool,
) -> None:
    """:func:`_check_reduced_kpoint_movements` for a wire or slab.

    The forces of a wire or slab are averaged over the group the input
    reader keeps (:func:`_wire_slab_symmetry`), which completes an
    irreducible wedge (``Monkhorst_Pack_Shift`` with a group, or an unequally
    weighted ``Kpoint_List``).  The bulk rules apply with that group and its
    lattice ``alatt``: a relaxation whose ``Move_Flag`` a group operation
    does not respect (the ionic loop keeps the respecting operations only)
    and molecular dynamics or ``manual`` movements with a weighted list are
    refused.  Molecular dynamics switches the group off (``parsec.F90:481``,
    ``ierr = 2``), so for that test the group is the one the structure would
    have without it; a slab without band structure, spin-orbit coupling or
    NSCF has none in PARSEC (its weighted list is a plain quadrature there,
    as with ``Ignore_Symmetry``).  ``Monkhorst_Pack_Shift`` with molecular
    dynamics is the wedge of time reversal alone (no group): accepted.
    """

    from ..Grid.space_group import confined_symmetry_lattice, find_wire_slab_space_group

    if ignore_symmetry or not (weighted_kpoints or monkhorst_pack_shift is not None):
        return
    lattice = confined_symmetry_lattice(
        periodic_cell.lattice_vectors, periodicity, grid.confinement_radius
    )
    group = space_group
    if molecular_dynamics and weighted_kpoints:
        if periodicity == 2 and not slab_symmetry_on:
            return
        try:
            group, _ = find_wire_slab_space_group(
                periodic_cell.lattice_vectors, periodicity, grid.confinement_radius, atoms
            )
        except ValueError as error:
            raise ParsecInputError(f"symmetry operations: {error}") from error
    _check_reduced_kpoint_movements(
        atoms,
        periodic_cell,
        group,
        movable=movable,
        relaxation=relaxation,
        molecular_dynamics=molecular_dynamics and weighted_kpoints,
        reduced=True,
        ignore_symmetry=ignore_symmetry,
        symmetry_lattice=lattice,
    )


def _save_wfn_bands(lines: tuple[str, ...]) -> tuple[int, ...]:
    """The state numbers of a ``Save_Wfn_Bands`` block (``usrinputfile.F90``).

    An empty block saves none; one line ``lo hi`` saves ``lo..hi``; one line
    with one number, or one number per line, saves those states.  PARSEC
    reads list-directed, so commas separate numbers too and numbers after
    the first two of a single line are ignored.
    """

    def numbers(line: str) -> list[str]:
        return [token for token in re.split(r"[\s,]+", line.strip()) if token]

    def integer(token: str) -> int:
        try:
            value = float(token.replace("d", "e").replace("D", "E"))
        except ValueError:
            raise ParsecInputError(
                'bad specification of "Save_Wfn_Bands" block: '
                f"{token!r} is not an integer"
            ) from None
        if value != int(value):
            raise ParsecInputError(
                f'bad specification of "Save_Wfn_Bands" block: {token!r} is not an integer'
            )
        return int(value)

    if not lines:
        return ()
    if len(lines) == 1:
        tokens = numbers(lines[0])
        if len(tokens) >= 2:
            low, high = integer(tokens[0]), integer(tokens[1])
            if high < low:
                raise ParsecInputError(
                    f"Save_Wfn_Bands range {low} {high} is empty; give 'first last' "
                    "with first <= last"
                )
            states = tuple(range(low, high + 1))
        elif tokens:
            states = (integer(tokens[0]),)
        else:
            states = ()
    else:
        states = []
        for line in lines:
            tokens = numbers(line)
            if not tokens:
                raise ParsecInputError('bad specification of "Save_Wfn_Bands" block')
            states.append(integer(tokens[0]))
        states = tuple(states)
    if any(value < 1 for value in states):
        raise ParsecInputError("Save_Wfn_Bands state numbers start at 1")
    return states


def _output_grid_data(value: str) -> tuple[str, ...]:
    """The fields of ``Output_Grid_Data`` (``usrinputfile.F90`` l.2598-2638).

    ``esdf_reduce`` lowers the case and removes dots and hyphens;
    ``esdf_split`` splits the words at commas (a semicolon starts a comment
    in this reader, and PARSEC notes that everything after one is ignored).
    ``none`` ends the list, ``all`` gives the four fields.
    """

    reduced = re.sub(r"[.\-]", "", value.lower()).strip()
    words = [word.strip() for word in reduced.split(",")][:10]
    fields: list[str] = []
    for word in words:
        if word == "none":
            break
        if word == "all":
            # PARSEC sets the first four flags and stops reading.
            return GRID_DATA_FIELDS
        if word not in GRID_DATA_FIELDS:
            raise ParsecInputError(f"Wrong input to output_grid_data: {word}")
        fields.append(word)
    return tuple(fields)


def _output_file_settings(scalar, blocks, one, optional_bool, has_kpoints: bool) -> OutputFileSettings:
    """``Output_Data`` and the other output-file keywords of ``usrinputfile.F90``."""

    keywords = {
        key
        for key in (
            "output_data",
            "output_all_states",
            "output_eigenvalues",
            "save_intermediate_eigenvalues",
            "save_intermediate_charge_density",
            "output_grid_data",
            "output_file_name",
        )
        if key in scalar
    }
    all_states = optional_bool("output_all_states")
    if all_states:
        selection = WavefunctionSelection(mode="all", explicit=True)
    elif "save_wfn_bands" in blocks:
        band_blocks = blocks["save_wfn_bands"]
        if len(band_blocks) != 1:
            raise ParsecInputError("provide a single Save_Wfn_Bands block")
        selection = WavefunctionSelection(
            mode="list", indices=_save_wfn_bands(band_blocks[0].value), explicit=True
        )
        keywords.add("save_wfn_bands")
    else:
        selection = WavefunctionSelection()
    name = one("output_file_name", "parsec.out").strip()
    if len(name) >= 2 and name[0] == name[-1] and name[0] in {"'", '"'}:
        name = name[1:-1].strip()
    if not name:
        raise ParsecInputError("Output_File_Name cannot be empty")
    return OutputFileSettings(
        output_data=optional_bool("output_data", True),
        wavefunctions=selection,
        intermediate_density=optional_bool("save_intermediate_charge_density"),
        # usrinputfile.F90 l.2645: eigen.dat is always written with k-points.
        eigenvalues=optional_bool("output_eigenvalues") or has_kpoints,
        all_states=all_states,
        intermediate_eigenvalues=(
            (optional_bool("output_eigenvalues") or has_kpoints)
            and optional_bool("save_intermediate_eigenvalues")
        ),
        grid_data=_output_grid_data(one("output_grid_data", "none")),
        output_file_name=name,
        keywords=frozenset(keywords),
    )


@dataclass(frozen=True)
class _PostProcessingInput:
    """What :func:`_post_processing_input` reads: NSCF, DOS, Calc_Hartree_G."""

    nscf: NSCFSettings | None
    dos: DOSSettings | None
    g_space_hartree: bool
    monkhorst_pack_dimensions: tuple[int, int, int] | None
    kpoints: np.ndarray | None
    kpoint_weights: np.ndarray | None
    # Output_GW with nscf_kpoints: nscf_kgrid, nscf_kgrid_shift and
    # nscf_fermi_level (Ry); ``None`` otherwise.
    gw_kgrid: tuple[int, int, int] | None = None
    gw_kshift: tuple[float, float, float] | None = None
    gw_fermi_level: float | None = None
    # Create_DOS with Restart_Run on Fortran's Monkhorst-Pack path: the shift
    # of the restarted SCF (``mpshift = dos_mpshift``), ``None``: unchanged.
    monkhorst_pack_shift: tuple[float, float, float] | None = None


#: The NSCF keys ``usrinputfile.F90`` reads only with ``Output_GW``.
_NSCF_GW_KEYS = ("nscf_kgrid", "nscf_kgrid_shift", "nscf_fermi_level")


def _nscf_gw_keys(scalar, blocks, one) -> tuple[tuple[int, int, int], tuple[float, float, float], float]:
    """``nscf_kgrid``, ``nscf_kgrid_shift`` and ``nscf_fermi_level`` (``usrinputfile.F90`` l.2492-2543).

    PARSEC reads them with ``Output_GW`` in a non-self-consistent run and
    stops (error 114) when one is missing; a block of more than one line is
    error 114 (kgrid) or 116 (shift), a negative shift error 117.  The grid
    must hold three integers of at least one (what BerkeleyGW's reader checks;
    PARSEC does not), the Fermi level is an ESDF physical value (default Ry).
    """

    grid_blocks = blocks.get("nscf_kgrid", [])
    if not grid_blocks:
        raise ParsecInputError(
            "Output_GW with nscf_kpoints: Have to specify nscf_kgrid (PARSEC error 114)"
        )
    if len(grid_blocks) != 1 or len(grid_blocks[0].value) != 1:
        raise ParsecInputError(
            "error in specifying kgrid for nscf calc: give one nscf_kgrid line "
            "(PARSEC error 114)"
        )
    values, unit = _block_numbers(grid_blocks[0].value, label="nscf_kgrid")
    if unit or len(values) != 3 or any(value != int(value) for value in values):
        raise ParsecInputError("nscf_kgrid: give three integers")
    kgrid = tuple(int(value) for value in values)
    if any(value < 1 for value in kgrid):
        raise ParsecInputError(
            "nscf_kgrid: every dimension must be at least 1 (BerkeleyGW refuses a "
            "smaller k-grid in the WFN header)"
        )
    shift_blocks = blocks.get("nscf_kgrid_shift", [])
    if not shift_blocks:
        raise ParsecInputError(
            "Output_GW with nscf_kpoints: Have to specify nscf_kgrid_shift (PARSEC error 114)"
        )
    if len(shift_blocks) != 1 or len(shift_blocks[0].value) != 1:
        raise ParsecInputError("error in specifying Monkhorst-Pack Shift (PARSEC error 116)")
    values, unit = _block_numbers(shift_blocks[0].value, label="nscf_kgrid_shift")
    if unit or len(values) != 3 or any(value < 0 for value in values):
        raise ParsecInputError("error in specifying Monkhorst-Pack Shift (PARSEC error 117)")
    kshift = tuple(float(value) for value in values)
    if "nscf_fermi_level" not in scalar:
        raise ParsecInputError(
            "ERROR: specification of Fermi level for non self consistent calculation "
            "not found! Please specify nscf_fermi_level taken from the scf calculation "
            "(PARSEC error 114)"
        )
    fermi_level = _esdf_energy_rydberg(one("nscf_fermi_level"), label="nscf_fermi_level")
    return kgrid, kshift, fermi_level


def _post_processing_input(
    scalar,
    blocks,
    one,
    optional_bool,
    *,
    is_periodic: bool,
    wire_or_slab: bool = False,
    periodicity: int = 3,
    periodic_cell: PeriodicCell | None,
    input_cell: PeriodicCell | None,
    restart_run: bool,
    states: int,
    symbols: list[str],
    monkhorst_pack_dimensions: tuple[int, int, int] | None,
    kpoints: np.ndarray | None,
    kpoint_weights: np.ndarray | None,
    spin_orbit: bool,
    warnings: list[str],
    output_gw: bool = False,
    noncollinear: bool = False,
    monkhorst_pack_shift: tuple[float, float, float] | None = None,
) -> _PostProcessingInput:
    """``nscf_*``, ``Create_DOS`` and its keys, ``Calc_Hartree_G`` (``usrinputfile.F90``).

    NSCF (l.1056-1118): ``nscf_kpoints`` (``kx ky kz w`` rows, ``w >= 0``,
    error 119; weights renormalized), ``nscf_kpoint_unit`` (the reciprocal
    lattice of the input cell ``lattvec``), ``nscf_states_num`` (default
    ``States_Num``), and ``Restart_Run`` (error 114).  ``nscf_kgrid``,
    ``nscf_kgrid_shift`` and ``nscf_fermi_level`` are read only with
    ``Output_GW`` (l.2493-2543, :func:`_nscf_gw_keys`), and are inert without.
    DOS (l.1122-1197, 1913): ``Create_DOS``, ``DOS_Pnum``, and for a cell
    ``DOS_MP_Grid``, ``DOS_MP_Shift``, ``YLM_DOS_L`` and the per-species
    ``ylm_dos_cutoff``; with ``Restart_Run`` the SCF becomes a Monkhorst-Pack
    run on the DOS grid (l.1188-1193), so the returned k-point sampling
    replaces the input's.  ``Calc_Hartree_G`` (l.907-912, bulk only).
    Keys PARSEC does not read in a given setting are accepted with a warning.
    """

    nscf_keys_given = sorted(
        {key for key in ("nscf_kpoint_unit", "nscf_states_num", "nscf_fermi_level") if key in scalar}
        | {f"begin {key}" for key in ("nscf_kpoints", "nscf_kgrid", "nscf_kgrid_shift") if key in blocks}
    )
    nscf = None
    gw_kgrid = gw_kshift = gw_fermi_level = None
    gw_keys = [key for key in _NSCF_GW_KEYS if key in scalar or key in blocks]
    if not is_periodic:
        if nscf_keys_given:
            warnings.append(
                f"{', '.join(nscf_keys_given)}: PARSEC reads the non-self-consistent "
                "keywords for periodic systems only; they have no effect here"
            )
    elif "nscf_kpoints" in blocks:
        nscf_blocks = blocks["nscf_kpoints"]
        if len(nscf_blocks) != 1:
            raise ParsecInputError("provide a single nscf_kpoints block")
        rows = []
        for line in nscf_blocks[0].value:
            values, unit = _block_numbers((line,), label="nscf_kpoints")
            if unit or len(values) != 4:
                raise ParsecInputError(
                    f"line {nscf_blocks[0].line}: every nscf_kpoints row must be "
                    "'kx ky kz weight' (four numbers, as usrinputfile.F90 reads it)"
                )
            rows.append(values)
        if not rows:
            raise ParsecInputError("the nscf_kpoints block is empty")
        array = np.asarray(rows, dtype=float)
        if np.any(array[:, 3] < 0.0):
            raise ParsecInputError(
                "nscf_kpoints: Sampling weight must be positive (PARSEC error 119)"
            )
        unit_name = _normalize_label(one("nscf_kpoint_unit", "cartesian_inverse_bohr"))
        try:
            k_points, weights = explicit_kpoints(array, unit=unit_name, cell=input_cell)
        except ValueError as error:
            raise ParsecInputError(f"nscf_kpoints: {error}") from error
        if wire_or_slab and np.any(np.abs(k_points[:, periodicity:]) > 1.0e-12):
            # calc_nscf.F90 would add -2i k.grad along a confined direction.
            raise ParsecInputError(
                "nscf_kpoints: a wire or slab has Bloch vectors along its periodic "
                "direction(s) only; the k-point components along the confined "
                "direction(s) must be zero"
            )
        number = _integer(one("nscf_states_num", str(states)), label="nscf_States_Num")
        if number < 1:
            raise ParsecInputError("nscf_States_Num must be positive")
        if not restart_run:
            raise ParsecInputError(
                "In case of non self consistent calculations please specify the "
                "restart flag (Restart_Run: .true.) to read the potential from a "
                "previous run (PARSEC error 114)"
            )
        if noncollinear:
            # A non-collinear parsec.dat (stype(2) 3 or 4) stores no direction
            # of the magnetization; restart_run.F90 restarts nothing from it.
            raise UnsupportedParsecOptionError(
                "nscf_kpoints with non-collinear magnetism is not implemented: the "
                "parsec.dat of a non-collinear run stores no magnetization direction "
                "(restart_run.F90 reads spin-orbit files only); a spin-orbit run "
                "(SO_from_scratch/SCF_SO without Non_Collinear_magnetism) is supported"
            )
        nscf = NSCFSettings(k_points=k_points, weights=weights, number_of_states=number)
        if output_gw:
            gw_kgrid, gw_kshift, gw_fermi_level = _nscf_gw_keys(scalar, blocks, one)
        elif gw_keys:
            warnings.append(
                f"{', '.join(gw_keys)}: PARSEC reads them only with Output_GW; they "
                "have no effect: the NSCF Fermi level is the one of the SCF that wrote "
                "parsec.dat"
            )
    elif nscf_keys_given:
        warnings.append(
            f"{', '.join(nscf_keys_given)} without a nscf_kpoints block: PARSEC reads "
            "them only for a non-self-consistent run; they have no effect"
        )

    # Create_DOS and DOS_Pnum are read in every run (l.1122-1124).
    create_dos = optional_bool("create_dos")
    points = _integer(one("dos_pnum", "1000"), label="DOS_Pnum")
    periodic_dos_keys = sorted(
        {f"begin {key}" for key in ("dos_mp_grid", "dos_mp_shift") if key in blocks}
        | {key for key in ("ylm_dos_l",) if key in scalar}
    )
    dos = None
    if create_dos and points < 1:
        raise ParsecInputError("DOS_Pnum must be positive")
    if not create_dos:
        given = periodic_dos_keys + (["dos_pnum"] if "dos_pnum" in scalar else [])
        if given:
            warnings.append(
                f"{', '.join(given)} without Create_DOS: .true. have no effect, as in PARSEC"
            )
    elif not is_periodic:
        if periodic_dos_keys:
            warnings.append(
                f"{', '.join(periodic_dos_keys)}: PARSEC reads them for periodic "
                "systems only; a cluster's DOS is the histogram of its occupations"
            )
        dos = DOSSettings(points=points)
    else:
        # A wire or slab: dos.F90 takes its Monkhorst-Pack grid (one point
        # along the confined directions) as a three-dimensional one, so the
        # tetrahedra are flat along those directions and the linear
        # interpolation runs along the periodic ones.
        if spin_orbit:
            raise UnsupportedParsecOptionError(
                "Create_DOS with spin-orbit coupling or non-collinear magnetism is not "
                "implemented (dos.F90 reads an undefined spin index for spinors)"
            )
        if nscf is not None:
            raise UnsupportedParsecOptionError(
                "Create_DOS together with nscf_kpoints is not implemented: PARSEC "
                "would build the tetrahedra of the SCF grid on the NSCF levels"
            )
        grid_blocks = blocks.get("dos_mp_grid", [])
        dos_grid = None
        if grid_blocks:
            if len(grid_blocks) != 1 or len(grid_blocks[0].value) != 1:
                raise ParsecInputError(
                    "error in specifying DOS MP Grid: give one DOS_MP_Grid block of one line"
                )
            values, unit = _block_numbers(grid_blocks[0].value, label="DOS_MP_Grid")
            if wire_or_slab and not unit and len(values) >= periodicity:
                # usrinputfile.F90:1134-1135 reads dos_mpgrid(1:per).
                values = list(values[:periodicity]) + [1.0] * (3 - periodicity)
            if (
                unit
                or len(values) != 3
                or any(value <= 0 or value != int(value) for value in values)
            ):
                raise ParsecInputError(
                    "error in specifying DOS MP Grid: three positive integers"
                )
            dos_grid = tuple(int(value) for value in values)
        dos_shift = None
        restart_shift = None
        if "dos_mp_shift" in blocks:
            if dos_grid is None:
                warnings.append(
                    "DOS_MP_Shift without DOS_MP_Grid is not read, as in PARSEC"
                )
            else:
                shift_blocks = blocks["dos_mp_shift"]
                if len(shift_blocks) != 1 or len(shift_blocks[0].value) != 1:
                    raise ParsecInputError("error in specifying DOS MP Shift")
                values, unit = _block_numbers(shift_blocks[0].value, label="DOS_MP_Shift")
                if wire_or_slab and not unit and len(values) >= periodicity:
                    values = list(values[:periodicity]) + [0.0] * (3 - periodicity)
                if unit or len(values) != 3 or any(value < 0 for value in values):
                    raise ParsecInputError("error in specifying DOS MP Shift")
                dos_shift = tuple(float(value) for value in values)
        manual = kpoints is not None
        if dos_grid is None and manual:
            if restart_run:
                raise UnsupportedParsecOptionError(
                    "Create_DOS with Restart_Run and a Kpoint_List needs DOS_MP_Grid: "
                    "PARSEC then runs the SCF on that Monkhorst-Pack grid"
                )
            warnings.append(
                "DOS cannot be created based on a manual MP grid, please specify a DOS "
                "MP grid. DOS creation aborted ! (as PARSEC)"
            )
        else:
            if dos_grid is None:
                dos_grid = monkhorst_pack_dimensions
            if restart_run:
                # usrinputfile.F90 l.1188-1193: the restarted SCF runs on the
                # Monkhorst-Pack grid of the DOS.
                if dos_grid is None:
                    raise UnsupportedParsecOptionError(
                        "Create_DOS with Restart_Run for a Gamma-point input needs "
                        "DOS_MP_Grid"
                    )
                if dos_shift is not None or monkhorst_pack_shift is not None:
                    # PARSEC's Monkhorst-Pack path (the shift block of the SCF
                    # or DOS_MP_Shift): mpshift = dos_mpshift, the block of
                    # DOS_MP_Shift, zero without it when DOS_MP_Grid is given,
                    # and the shift of the SCF without DOS_MP_Grid.  Without
                    # either block this port's grid of the DOS divisions is
                    # used, as for every input without a shift block.
                    if "dos_mp_grid" in blocks:
                        restart_shift = dos_shift if dos_shift is not None else (0.0, 0.0, 0.0)
                    else:
                        restart_shift = monkhorst_pack_shift
                    dos_shift = restart_shift
                monkhorst_pack_dimensions = dos_grid
                kpoints = None
                kpoint_weights = None
            elif monkhorst_pack_dimensions is None:
                raise UnsupportedParsecOptionError(
                    "Create_DOS for a periodic cell needs the Monkhorst-Pack k-points of "
                    "the SCF (Kpoint_Method: mp), or DOS_MP_Grid with Restart_Run (PARSEC "
                    "stops: 'DOS: kpoint map is not set up!')"
                )
            else:
                if dos_shift is not None:
                    warnings.append(
                        "DOS_MP_Shift has no effect without Restart_Run: the DOS uses the "
                        "k-points of the SCF, as in PARSEC"
                    )
                if dos_grid != monkhorst_pack_dimensions:
                    warnings.append(
                        "You are creating the DOS using the original MP Grid; in order to "
                        "use the DOS grid perform a restart run with the same input (as "
                        "PARSEC)"
                    )
                dos_grid = monkhorst_pack_dimensions
            ylm_l = _integer(one("ylm_dos_l", "-1"), label="YLM_DOS_L")
            if ylm_l > 3:
                raise UnsupportedParsecOptionError(
                    "YLM_DOS_L above 3: dos.F90 has the real harmonics s, p, d and f only"
                )
            ylm_l = max(ylm_l, -1)
            occurrences = scalar.get("ylm_dos_cutoff", [])
            cutoffs = []
            for index, symbol in enumerate(symbols):
                text = str(occurrences[index].value) if index < len(occurrences) else "1.0"
                radius = _physical_length(text, label=f"{symbol} ylm_dos_cutoff")
                if not radius > 0.0:
                    raise ParsecInputError(f"Atom_Type {symbol}: ylm_dos_cutoff must be positive")
                cutoffs.append((symbol, radius))
            dos = DOSSettings(
                points=points,
                monkhorst_pack_grid=dos_grid,
                monkhorst_pack_shift=dos_shift,
                ylm_l=ylm_l,
                ylm_cutoffs=tuple(cutoffs),
                fortran_projection=optional_bool("ylm_dos_fortran_projection"),
            )
    if "ylm_dos_fortran_projection" in scalar and (dos is None or dos.ylm_l < 0):
        warnings.append("YLM_DOS_Fortran_Projection without a projected DOS has no effect")

    g_space = False
    if optional_bool("calc_hartree_g"):
        # usrinputfile.F90:907 reads it in the bulk case (pbc%per = 3) only.
        if not is_periodic or wire_or_slab:
            warnings.append(
                "Calc_Hartree_G is read for a bulk cell only, as in PARSEC; no effect here"
            )
        else:
            g_space = True
            warnings.append(
                "Calc_Hartree_G: PARSEC's hpot_G is broken (it prints 'IT WILL NOT "
                "WORK': it reads the star-averaged density with the index of each "
                "G-vector, and its Hartree energy is not used); the Python port solves "
                "the intended V_H(G) = 8 pi rho(G)/|G|^2 exactly by FFT "
                "(parsec_python.Hartree.gspace). The default finite-difference solve "
                "is unchanged without the keyword"
            )
    return _PostProcessingInput(
        nscf=nscf,
        dos=dos,
        g_space_hartree=g_space,
        monkhorst_pack_dimensions=monkhorst_pack_dimensions,
        kpoints=kpoints,
        kpoint_weights=kpoint_weights,
        gw_kgrid=gw_kgrid,
        gw_kshift=gw_kshift,
        gw_fermi_level=gw_fermi_level,
        monkhorst_pack_shift=restart_shift if create_dos and is_periodic else None,
    )


def _gw_settings(
    post_processing: _PostProcessingInput,
    *,
    atoms,
    periodic_cell: PeriodicCell | None,
    is_periodic: bool,
    wire_or_slab: bool,
    spinors: bool,
    bands: bool,
    moves: bool,
    forces: bool,
    monkhorst_pack_dimensions,
    monkhorst_pack_shift,
    explicit_kpoints: bool,
    space_group,
    ignore_symmetry: bool,
    warnings: list[str],
) -> GWSettings:
    """``Output_GW``: what ``gw_write`` needs, and the cases refused by name.

    PARSEC calls ``gw_write`` after the SCF or the NSCF run (``parsec.F90``
    l.2095-2147) with the full space group (``symm``; the identity with
    ``Ignore_Symmetry``).  With ``nscf_kpoints`` the WFN header carries
    ``nscf_kgrid``/``nscf_kgrid_shift``; after an SCF PARSEC writes the
    never-set ``nscf%kgrid`` (``0 0 0``), which BerkeleyGW's reader refuses:
    this port writes the Monkhorst-Pack grid of the SCF there (shift in
    BerkeleyGW's convention ``k = (i + s)/n``), and ``0 0 0`` for a
    ``Kpoint_List`` (with a warning).
    """

    from ..Grid.space_group import find_space_group, identity_space_group

    if not is_periodic:
        raise UnsupportedParsecOptionError(
            "Output_GW is implemented for a bulk periodic cell only: gw_write needs the "
            "reciprocal space of the cell (PARSEC's cluster branch reads periodic data "
            "that a cluster run never sets)"
        )
    if wire_or_slab:
        raise UnsupportedParsecOptionError(
            "Output_GW with Boundary_Conditions wire or slab is not implemented: "
            "gw_write maps the grid onto a three-dimensional FFT box, which the confined "
            "grid of a wire or slab does not fill"
        )
    if spinors:
        raise UnsupportedParsecOptionError(
            "Output_GW with spin-orbit coupling or non-collinear magnetism is not "
            "implemented: gw_write writes no two-component spinors"
        )
    if bands:
        raise UnsupportedParsecOptionError(
            "Output_GW together with bandstruc is not implemented: PARSEC's calc_bands "
            "replaces the k-points and levels before gw_write writes them"
        )
    if moves or forces:
        raise UnsupportedParsecOptionError(
            "Output_GW with Minimization, Molecular_Dynamics or forces (Skip_Force: "
            "false) is not implemented: run it as a single point of the final structure"
        )
    nscf = post_processing.nscf
    if nscf is None and monkhorst_pack_dimensions is None and not explicit_kpoints:
        raise UnsupportedParsecOptionError(
            "Output_GW after a Gamma-point SCF is not implemented: PARSEC writes no "
            "k-point (nk = 0), which BerkeleyGW cannot read; use Kpoint_Method mp with "
            "a 1 1 1 grid, or a non-self-consistent run (nscf_kpoints)"
        )
    if space_group is None:
        if ignore_symmetry:
            space_group = identity_space_group()
        else:
            try:
                space_group = find_space_group(periodic_cell.lattice_vectors, atoms)
            except ValueError as error:
                raise ParsecInputError(f"symmetry operations: {error}") from error
    if nscf is not None:
        return GWSettings(
            space_group=space_group,
            kgrid=post_processing.gw_kgrid,
            kshift=post_processing.gw_kshift,
            fermi_level=post_processing.gw_fermi_level,
        )
    if monkhorst_pack_dimensions is not None:
        dimensions = tuple(int(value) for value in monkhorst_pack_dimensions)
        if monkhorst_pack_shift is not None:
            shift = tuple(float(value) for value in monkhorst_pack_shift)
        else:
            # This port's grid (2i - n - 1)/(2n) is k = (i + s)/n with s = 1/2
            # for even n and s = 0 for odd n, modulo a reciprocal lattice vector.
            shift = tuple(0.5 if value % 2 == 0 else 0.0 for value in dimensions)
        warnings.append(
            "Output_GW after an SCF: the WFN header gets the Monkhorst-Pack grid "
            f"{dimensions} and shift {shift} of the SCF (PARSEC writes 0 0 0, which "
            "BerkeleyGW's reader refuses)"
        )
        return GWSettings(space_group=space_group, kgrid=dimensions, kshift=shift)
    warnings.append(
        "Output_GW after an SCF on a Kpoint_List: the WFN header gets the k-grid 0 0 0 "
        "as in PARSEC; BerkeleyGW's reader refuses it unless told not to check the grid"
    )
    return GWSettings(space_group=space_group)


def _point_charges(
    scalar: dict[str, list["_InputItem"]],
    blocks: dict[str, list["_InputItem"]],
    enabled: bool,
    is_periodic: bool,
    coordinate_factor: float,
    warnings: list[str],
) -> tuple[tuple[PointCharge, ...], tuple[int, ...]]:
    """The point charges of ``usrinputfile.F90`` (lines 2143-2220).

    ``Add_Point_Charges`` (false), ``Point_Typ_Num`` types, and for the n-th
    type the n-th ``Pt_Chg`` (in e, zero where it is missing) and the n-th
    ``Point_Coord`` block (``x y z`` rows in the unit of ``Coordinate_Unit``).
    Returns the charges and the number of charges of each type.
    """

    given = [key for key in ("point_typ_num", "pt_chg") if key in scalar]
    if "point_coord" in blocks:
        given.append("begin Point_Coord")
    if not enabled:
        if given:
            warnings.append(
                f"{', '.join(given)} without Add_Point_Charges: .true. are ignored, "
                "as in PARSEC"
            )
        return (), ()
    if is_periodic:
        # usrinputfile.F90, error 145.
        raise UnsupportedParsecOptionError(
            "point charges (Add_Point_Charges) are only available in "
            "non-periodic systems"
        )
    occurrences = scalar.get("point_typ_num", [])
    types = (
        0
        if not occurrences
        else _integer(str(occurrences[0].value), label="Point_Typ_Num")
    )
    if types <= 0:
        # usrinputfile.F90, error 146 (0 is ESDF's default).
        raise ParsecInputError(
            "unknown number of point charge types: Point_Typ_Num is missing"
        )
    charge_items = scalar.get("pt_chg", [])
    coordinate_blocks = blocks.get("point_coord", [])
    if len(coordinate_blocks) < types:
        # usrinputfile.F90, error 148.
        raise ParsecInputError(
            f"no point charge coordinates were found for point charge type "
            f"{len(coordinate_blocks) + 1}: give one Point_Coord block per type"
        )
    for label, count in (("Pt_Chg", len(charge_items)), ("Point_Coord", len(coordinate_blocks))):
        if count > types:
            warnings.append(
                f"{count} {label} entries for Point_Typ_Num {types}: the "
                f"last {count - types} are not read, as in PARSEC"
            )
    charges: list[PointCharge] = []
    counts: list[int] = []
    for index in range(types):
        charge = (
            _charge_electrons(str(charge_items[index].value), label="Pt_Chg")
            if index < len(charge_items)
            else 0.0
        )
        item = coordinate_blocks[index]
        if not item.value:
            raise ParsecInputError(
                f"line {item.line}: the Point_Coord block of type {index + 1} is empty"
            )
        for row in item.value:
            values, unit = _block_numbers((row,), label="Point_Coord")
            if unit or len(values) < 3:
                raise ParsecInputError(
                    f"line {item.line}: every Point_Coord row must contain three "
                    "coordinates"
                )
            charges.append(
                PointCharge(charge, np.asarray(values[:3], dtype=float) * coordinate_factor)
            )
        counts.append(len(item.value))
    return tuple(charges), tuple(counts)


# --- FDET (PARSEC 1.5 frozen-density embedding) --------------------------------

#: The FDET keywords of ``usrinputfile.F90:2190-2232`` (esdf_key 191-198) and
#: the PARSEC.py ones of this port.
FDET_KEYWORDS = (
    "potential_field",
    "potential_field_name",
    "kinetic_energy_functional",
    "pbe_m",
    "pbe_k",
    "boundary_sphere_around_tip",
    # PARSEC.py extensions: the form of the kinetic potentials (the corrected
    # derivatives by default, PARSEC 1.5's for comparisons) and pot.dat.
    "fdet_kinetic_form",
    "write_potential_field",
    # PARSEC.py extension: the AFM scan of an FDET tip (parsec_python.Embedding.afm).
    "afm_scan",
    "afm_scan_heights",
    "afm_scan_height_step",
    "afm_scan_line_range",
)
#: The block of the lateral points of an AFM scan.
FDET_BLOCKS = ("afm_scan_lateral_points",)
#: ``esdf_reduce`` of ``kinetic_energy_functional`` -> the functional (usrinputfile.F90:2203-2227).
_FDET_FUNCTIONALS = {"pw": "pw", "pb": "pb", "tf": "tf", "tw": "tw", "no": "none"}
_FDET_FORMS = {"corrected": "corrected", "parsec15": "parsec1.5", "15": "parsec1.5", "fortran": "parsec1.5"}


def _esdf_reduce(value: str) -> str:
    """``esdf_reduce`` of the first word: lower case, ``.`` and ``-`` removed."""

    words = _unquoted(value).split()
    word = words[0] if words else ""
    return word.lower().replace(".", "").replace("-", "")


@dataclass(frozen=True)
class _FDETInput:
    """What :func:`_fdet_settings` reads: the embedding of the problem and the pot.dat switch."""

    embedding: EmbeddingSettings | None = None
    write_potential_field: bool = False


def _fdet_settings(
    scalar,
    one,
    optional_bool,
    source: Path,
    warnings: list[str],
    *,
    is_periodic: bool,
    scf,
    point_charges,
    radius_given: bool,
    polarizability: bool,
    energy_refinement,
    nscf,
    specifications,
    ignore_symmetry: bool,
    minimization: str,
    molecular_dynamics: bool,
    extrapolation: str,
    grid_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> _FDETInput:
    """``Potential_Field`` and its keywords (PARSEC 1.5, ``usrinputfile.F90:2190-2232``).

    ``Potential_Field`` (false) turns FDET on; then ``Potential_Field_Name``
    (``pot.dat``, beside the input) names the sample file,
    ``Kinetic_Energy_Functional`` (``tf``) the kinetic functional (``pw``,
    ``pb``, ``tf``, ``tw``, ``no``; any other word gives TF, as in 1.5, with a
    warning), ``pbe_m``/``pbe_k`` (0.2319, 0.8438) the parameters of ``pb``.
    ``FDET_Kinetic_Form`` (PARSEC.py: ``corrected``, the default, or
    ``parsec1.5``) chooses the kinetic potentials of
    :mod:`parsec_python.Embedding.kinetic`.  ``Write_Potential_Field``
    (PARSEC.py, false) writes ``pot.dat`` after the run, which PARSEC 1.5 does
    after every run.  The combinations FDET does not cover here are refused.
    """

    write_field = optional_bool("write_potential_field")
    if write_field:
        if is_periodic:
            raise UnsupportedParsecOptionError(
                "Write_Potential_Field (pot.dat) is implemented for clusters only"
            )
        if scf.spin_polarized or scf.self_consistent_spin_orbit or scf.noncollinear:
            raise UnsupportedParsecOptionError(
                "Write_Potential_Field (pot.dat) is implemented for spin-unpolarized runs without "
                "spin-orbit coupling only"
            )
    if not optional_bool("potential_field"):
        ignored = [key for key in FDET_KEYWORDS[1:7] if key in scalar]
        if ignored:
            warnings.append(
                f"{', '.join(ignored)} without Potential_Field: .true. are ignored, as in PARSEC 1.5"
            )
        return _FDETInput(write_potential_field=write_field)

    raw_name = _unquoted(one("potential_field_name", "pot.dat")).strip() or "pot.dat"
    field_path = Path(raw_name).expanduser()
    if not field_path.is_absolute():
        field_path = source.parent / field_path
    field_path = field_path.resolve()
    reduced = _esdf_reduce(one("kinetic_energy_functional", "tf"))
    functional = _FDET_FUNCTIONALS.get(reduced)
    if functional is None:
        # usrinputfile.F90:2222-2225: "No kinetic energy functional has been specified".
        warnings.append(
            f"Kinetic_Energy_Functional {one('kinetic_energy_functional')!r} is not one of pw, pb, tf, "
            "tw, no: the Thomas-Fermi functional is used, as in PARSEC 1.5"
        )
        functional = "tf"
    mu, kappa = 0.2319, 0.8438
    if functional == "pb":
        mu = _fortran_float(one("pbe_m", "0.2319"), label="pbe_m")
        kappa = _fortran_float(one("pbe_k", "0.8438"), label="pbe_k")
        if kappa == 0.0:
            raise ParsecInputError("pbe_k (kappa of the PBE-like kinetic functional) must not be zero")
    elif "pbe_m" in scalar or "pbe_k" in scalar:
        warnings.append("pbe_m and pbe_k are read with Kinetic_Energy_Functional pb only, as in PARSEC 1.5")
    form_word = _esdf_reduce(one("fdet_kinetic_form", "corrected"))
    form = _FDET_FORMS.get(form_word)
    if form is None:
        raise ParsecInputError("FDET_Kinetic_Form must be corrected or parsec1.5")
    if "fdet_kinetic_form" in scalar and functional in ("tf", "none"):
        warnings.append(
            "FDET_Kinetic_Form changes the tw, pw and pb potentials only; TF and no correction "
            "are the same in both forms"
        )
    tip_sphere = optional_bool("boundary_sphere_around_tip")
    if tip_sphere:
        # fdet_tipsphere.F90:217-220: the tip grid must be unshifted (622): Ignore_Symmetry without a Grid_Shift
        # block, or a Grid_Shift block of zeros (the spot field's own shift is checked when it is read, 625).
        if any(abs(value) > 1.0e-10 for value in grid_shift):
            raise ParsecInputError(
                "Boundary_Sphere_Around_Tip needs an unshifted tip grid (PARSEC error 622): the grid shift is "
                f"{list(grid_shift)}; give Ignore_Symmetry: true and no Grid_Shift block (or Grid_Shift 0 0 0)"
            )
        if minimization in ("bfgs", "simple") or molecular_dynamics:
            # PARSEC 1.5 recentres the tip after every move as if domove gave spot-frame coordinates,
            # which only manual.dat does.
            raise UnsupportedParsecOptionError(
                "Boundary_Sphere_Around_Tip is implemented for single points and Minimization: manual "
                "(scans) only"
            )

    # What the FDET of this port covers: a spin-unpolarized cluster tip.
    refusals = []
    if is_periodic:
        refusals.append("a periodic Boundary_Conditions (the slab tip runs of PARSEC 1.5)")
    if scf.spin_polarized:
        # PARSEC 1.5 adds no rho_A and no T_nad with spin (hartree.F90:1886-1892).
        refusals.append("Spin_Polarization")
    if scf.self_consistent_spin_orbit or scf.noncollinear or any(
        specification.spin_orbit for specification in specifications.values()
    ):
        refusals.append("spin-orbit coupling or non-collinear magnetism")
    if scf.symmetrize_density:
        refusals.append("Symmetrize_Charge_Density (the frozen field has no point group)")
    if scf.van_der_waals:
        refusals.append("Van_Der_Waals")
    if any(getattr(specification, "has_hubbard", False) for specification in specifications.values()):
        refusals.append("LDA+U")
    if polarizability:
        refusals.append("Polarizability")
    if energy_refinement is not None:
        refusals.append("Energy_Refinement")
    if nscf is not None:
        refusals.append("a non-self-consistent run (nscf_kpoints)")
    if extrapolation != "none":
        refusals.append("Potential_Extrapolation (a moved FDET tip starts as PARSEC 1.5 starts it)")
    if refusals:
        raise UnsupportedParsecOptionError(
            "FDET (Potential_Field) is not implemented together with " + ", ".join(refusals)
        )
    if not radius_given:
        raise ParsecInputError(
            "FDET (Potential_Field) needs Boundary_Sphere_Radius: the tip run uses the grid of the sample run"
        )
    if write_field and field_path == (source.parent / "pot.dat").resolve():
        # PARSEC 1.5 overwrites the sample file it reads.
        raise ParsecInputError(
            f"Write_Potential_Field would overwrite the sample field {field_path} that this run reads; "
            "give the sample file another Potential_Field_Name"
        )
    if not point_charges:
        warnings.append(
            "FDET without Add_Point_Charges: the ion-ion energy and forces have no sample ions (PARSEC 1.5 "
            "takes them as point charges of the tip run)"
        )
    return _FDETInput(
        embedding=EmbeddingSettings(
            field_path=field_path,
            kinetic_functional=functional,
            pbe_mu=mu,
            pbe_kappa=kappa,
            kinetic_form=form,
            tip_sphere=tip_sphere,
        ),
        write_potential_field=write_field,
    )


def _afm_scan_settings(scalar, blocks, one, optional_bool, *, embedding, coordinate_factor: float, spacing: float,
                       minimization: str, molecular_dynamics: bool, atoms):
    """``AFM_Scan`` (PARSEC.py): the scan of an FDET tip, ``None`` without it.

    ``AFM_Scan_Heights`` (3), ``AFM_Scan_Height_Step`` (the grid spacing),
    the block ``AFM_Scan_Lateral_Points`` (rows ``dx dy`` in the
    ``Coordinate_Unit``; one point ``0 0`` without it) and
    ``AFM_Scan_Line_Range`` (``first last``, 1-based; all lines without it).
    The scan runs as ``Minimization: manual`` over the structures of
    :class:`parsec_python.Embedding.afm.AFMScanSettings`.
    """

    from ..Embedding.afm import AFMScanSettings

    given = [key for key in FDET_KEYWORDS[-3:] if key in scalar] + [f"begin {key}" for key in FDET_BLOCKS if key in blocks]
    if not optional_bool("afm_scan"):
        if given:
            raise ParsecInputError(f"{', '.join(given)} need AFM_Scan: true")
        return None
    if embedding is None:
        raise ParsecInputError("AFM_Scan scans an FDET tip: it needs Potential_Field: true")
    if minimization != "none" or molecular_dynamics:
        raise UnsupportedParsecOptionError(
            "AFM_Scan moves the tip itself: Minimization and Molecular_Dynamics are not taken together with it"
        )
    heights = _integer(one("afm_scan_heights", "3"), label="AFM_Scan_Heights")
    step = (
        _physical_length(one("afm_scan_height_step"), label="AFM_Scan_Height_Step")
        if "afm_scan_height_step" in scalar
        else float(spacing)
    )
    points = [(0.0, 0.0)]
    if "afm_scan_lateral_points" in blocks:
        points = []
        for item in blocks["afm_scan_lateral_points"]:
            for row in item.value:
                values, unit = _block_numbers((row,), label="AFM_Scan_Lateral_Points")
                if unit or len(values) < 2:
                    raise ParsecInputError(f"line {item.line}: every AFM_Scan_Lateral_Points row needs dx dy")
                points.append((values[0] * coordinate_factor, values[1] * coordinate_factor))
    line_range = None
    if "afm_scan_line_range" in scalar:
        words = _unquoted(one("afm_scan_line_range")).split()
        if len(words) != 2:
            raise ParsecInputError("AFM_Scan_Line_Range needs two line numbers: first last")
        line_range = (_integer(words[0], label="AFM_Scan_Line_Range"), _integer(words[1], label="AFM_Scan_Line_Range"))
    try:
        return AFMScanSettings(
            heights=heights,
            height_step=step,
            lateral_points=tuple(points),
            line_range=line_range,
            tip_positions=tuple(tuple(float(value) for value in atom.position) for atom in atoms),
        )
    except ValueError as error:
        raise ParsecInputError(str(error)) from error


def _check_scan_structures(afm_scan, structures, atoms, grid, *, tip_sphere: bool) -> None:
    """Refuse an ``AFM_Scan`` one of whose tip positions puts an atom outside the domain.

    The run checks every structure when it starts its SCF loop (``isolat.f90``, ``_require_atoms_inside_domain``
    of :mod:`parsec_python.SCF.single_point`); a scan would stop there after the points before it.  The same test
    here, on the positions the run uses: those of the scan in the classical mode, those in the tip sphere
    (moved by ``-S h``, :func:`parsec_python.Embedding.tip_sphere.tip_offset`) with the tip sphere.
    """

    from ..Grid.cluster import _inside_domain

    spacing = float(getattr(grid, "spacing", 0.0) or 0.0)
    for (line, height, displacement), positions in zip(afm_scan.offsets(), structures):
        used = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if tip_sphere and spacing > 0.0:
            from ..Embedding.tip_sphere import tip_offset

            used = used - tip_offset(used, spacing).astype(np.float64) * spacing
        inside = _inside_domain(grid, used)
        if not np.all(inside):
            index = int(np.flatnonzero(~inside)[0])
            raise ParsecInputError(
                f"AFM_Scan: the tip position of line {line + 1}, height {height + 1} (displacement "
                f"{', '.join(f'{float(value):.6f}' for value in displacement)} bohr) puts atom {atoms[index].symbol} "
                f"at {np.round(used[index], 6).tolist()} outside the domain"
                + (" of the tip sphere" if tip_sphere else "")
                + ": enlarge Boundary_Sphere_Radius or shorten the scan (AFM_Scan_Lateral_Points, AFM_Scan_Heights, "
                "AFM_Scan_Height_Step)"
            )


def _check_hubbard_and_dispersion(
    specifications,
    *,
    van_der_waals: bool,
    is_periodic: bool,
    spin_polarized: bool,
    kpoints: bool,
    spin_orbit: bool,
    bands: bool,
    energy_refinement: bool,
    forces: bool,
) -> None:
    """Refuse LDA+U and TS combinations the drivers do not implement.

    LDA+U runs in the scalar loops (isolated, isolated spin, periodic Gamma,
    periodic spin), the scalar and collinear-spin k-point loops (with the
    Bloch projectors this port uses in place of PARSEC's defective k-point
    form, :mod:`parsec_python.SCF.kpoint_hubbard`) and the band structures
    of their converged Hamiltonians; the TS correction after every loop (the
    spin-orbit and non-collinear ones included), and ``Energy_Refinement``
    (isolated systems) refines the energy that holds them.  LDA+U with
    spin-orbit coupling or non-collinear magnetism is refused (PARSEC's path
    is not a spinor LDA+U, see below).  Forces with ``ldaplusu_j`` are
    refused: PARSEC's J term is not the derivative of its energy, and its
    LDA+U forces are zero; forces with LDA+U and k-points likewise (the
    Bloch projector gradients are not implemented).
    """

    hubbard = any(specification.has_hubbard for specification in specifications.values())
    for active, name in ((hubbard, "LDA+U (ldaplusu_u/ldaplusu_j)"), (van_der_waals, "Van_Der_Waals")):
        if not active:
            continue
        refused = [
            label
            for condition, label in (
                # PARSEC's spin-orbit path builds N from the up components of
                # the spinors alone (calcden.f90z) and applies the matrix of one
                # channel to both components (zmatvec_so -> plusu): there is
                # no spinor LDA+U to port.
                (
                    spin_orbit and name.startswith("LDA"),
                    "spin-orbit coupling or non-collinear magnetism",
                ),
            )
            if condition
        ]
        if refused:
            raise UnsupportedParsecOptionError(
                f"{name} is not implemented together with {', '.join(refused)}"
            )
    if forces and any(
        specification.hubbard_j != 0.0 for specification in specifications.values()
    ):
        raise UnsupportedParsecOptionError(
            "forces, relaxation and molecular dynamics with ldaplusu_j are not "
            "implemented: PARSEC's J term is not the derivative of its LDA+U "
            "energy (and PARSEC's LDA+U forces are zero); give ldaplusu_j 0"
        )
    if forces and hubbard and kpoints:
        raise UnsupportedParsecOptionError(
            "forces, relaxation and molecular dynamics with LDA+U and k-point "
            "sampling are not implemented (PARSEC's LDA+U forces are zero)"
        )




def _default_routines_periodic_refusal(
    periodic_cell,
    grid,
    atoms,
    *,
    slab: bool,
    kpoints,
    kpoint_weights,
    monkhorst_pack_dimensions,
    spin_polarized: bool,
    ionic: bool,
    functional: str = "ca",
) -> str | None:
    """What Andrew Grieder's routines refuse of a periodic input his parser takes, as a reason, or ``None``.

    Unequal k-point weights of a slab, or with forces, relaxation or dynamics (a bulk single point takes them as
    given, :func:`_default_routines_list_symmetry`); a slab with forces, relaxation or dynamics (his slab has energies
    only); collinear spin
    with k-points other than Gamma and forces (his CLI); a grid his rule refuses (an axis of ``n <= 2 M`` points, ``M``
    the stencil half width, with Fortran's ``adjust_step`` for a bulk cell; fewer than ``2 M`` in the plane of a slab
    with ``2 nint(L/2h)``; a side of 1000 steps or more).
    """

    from ..Grid.pbc import fortran_points_per_axis

    if (
        (ionic or slab)
        and kpoint_weights is not None
        and not np.allclose(kpoint_weights, kpoint_weights[0], rtol=1.0e-12, atol=0.0)
    ):
        # A single point of a bulk cell takes the weights as given since 2026-11-08 (_default_routines_list_symmetry);
        # the forces of a weighted list (a wedge) are completed over the space group by the Fortran-derived routines.
        return (
            "a Kpoint_List with unequal weights"
            if slab
            else "forces, relaxation or molecular dynamics with a Kpoint_List of unequal weights"
        )
    if slab and ionic:
        return "forces, relaxation or molecular dynamics of a slab"
    gamma_only = (
        (kpoints is None and monkhorst_pack_dimensions is None)
        or (monkhorst_pack_dimensions is not None and tuple(monkhorst_pack_dimensions) == (1, 1, 1))
        or (kpoints is not None and len(kpoints) == 1 and not np.any(kpoints))
    )
    if spin_polarized and ionic and not gamma_only:
        return "forces, relaxation or molecular dynamics of a spin-polarized cell with k-points"
    if spin_polarized and ionic and functional == "pbe":
        # his spin loop of a cell evaluates pbe_spin, which takes no periodic grid there (it stops)
        return "forces, relaxation or molecular dynamics of a spin-polarized PBE cell"
    half_width = grid.stencil_half_width
    lattice = np.asarray(periodic_cell.lattice_vectors, dtype=float)
    if slab:
        in_plane = (2 * np.round(np.linalg.norm(lattice[:2], axis=1) / (2.0 * grid.spacing))).astype(np.int64)
        if np.any(in_plane < 2 * half_width):
            return f"a slab grid of {in_plane.tolist()} in-plane points (fewer than {2 * half_width})"
        return None
    try:
        counts = fortran_points_per_axis(np.linalg.norm(lattice, axis=1), grid.spacing)
    except ValueError as error:
        return f"a grid the default rule cannot build ({error})"
    if np.any(counts <= 2 * half_width):
        return f"a grid of {counts.tolist()} points (the default rule needs more than {2 * half_width} per axis)"
    return None


def _gamma_only_sampling(monkhorst_pack_dimensions, explicit_kpoint_array) -> bool:
    """Whether a k-point input samples the Gamma point alone (``Monkhorst_Pack_Grid 1 1 1``, a list of k = 0)."""
    if monkhorst_pack_dimensions is not None:
        return explicit_kpoint_array is None and tuple(int(n) for n in monkhorst_pack_dimensions) == (1, 1, 1)
    if explicit_kpoint_array is None:
        return False
    points = np.asarray(explicit_kpoint_array, dtype=float)
    return points.shape[0] == 1 and not np.any(points)


def _small15_forces_requested(scalar, optional_bool) -> bool:
    """Whether the input asks for the forces of a single point.

    ``Skip_Force: .false.`` (this port's keyword) or PARSEC 1.5's
    ``Calculate_Force: .true.`` (``usrinputfile.F90:4151``).  PARSEC computes
    the forces of every run (1.5's default ``Calculate_Force`` is true); this
    port keeps its default, forces on request (kept on 2026-11-04), so only a
    ``Calculate_Force`` written in the input gives 1.5's behaviour.
    """

    if "calculate_force" in scalar and optional_bool("calculate_force"):
        return True
    return "skip_force" in scalar and not optional_bool("skip_force")


def _small15_ionic_keywords(scalar, optional_bool, *, atoms_move: bool, warnings: list[str]) -> bool:
    """Check PARSEC 1.5's force keywords; return :func:`_small15_forces_requested`.

    * ``Calculate_Force`` (L): with ``Skip_Force`` both must agree.  ``.false.``
      with a relaxation or molecular dynamics is refused: 1.5 would move the
      atoms with the ion-ion forces alone (``parsec.F90:2422-2505`` skips the
      local and nonlocal forces), which this port does not implement.
    * ``Use_Old_Forces`` (L): 1.5's serial force routines
      (``forcloc_old``/``forcnloc_old``, ``parsec.F90:2436,2477``), the same
      formulas as the default ones; accepted, no effect here.
    """

    if "calculate_force" in scalar:
        calculate = optional_bool("calculate_force")
        if "skip_force" in scalar and optional_bool("skip_force") == calculate:
            raise ParsecInputError(
                "Calculate_Force and Skip_Force contradict each other; give one of them"
            )
        if not calculate and atoms_move:
            raise UnsupportedParsecOptionError(
                "Calculate_Force: .false. with a relaxation or molecular dynamics is not "
                "implemented (PARSEC 1.5 would move the atoms with the ion-ion forces alone)"
            )
    if "use_old_forces" in scalar and optional_bool("use_old_forces"):
        warnings.append(
            "Use_Old_Forces selects PARSEC 1.5's serial force routines, which compute "
            "the same forces; it changes nothing here"
        )
    return _small15_forces_requested(scalar, optional_bool)


def _routines_choice_matters(scalar, one, optional_bool) -> bool:
    """Whether the two sets of routines (parsec_python.Input.routines) can differ for this input.

    A periodic cell, or the forces, relaxation or dynamics of spinor orbitals (``SO_from_scratch``/``SCF_SO``,
    ``Non_Collinear_magnetism``).  Every other calculation runs the same code under both.
    """

    boundary = _normalize_label(one("boundary_conditions", "cluster"))
    if boundary not in {"cluster", "0d"}:
        return True
    spinor = any(
        key in scalar and optional_bool(key)
        for key in ("so_from_scratch", "scf_so", "non_collinear_magnetism")
    )
    if not spinor:
        return False
    moves = _normalize_label(one("minimization", "none")) not in {"none", ""}
    dynamics = "molecular_dynamics" in scalar and optional_bool("molecular_dynamics")
    forces = _small15_forces_requested(scalar, optional_bool)
    return bool(moves or dynamics or forces)


#: Values of the labels Andrew Grieder's parser (or his SCF drivers) refuse although the labels are his
#: (qian/pbc_forces and qian/slab_boundary).
_DEFAULT_CORRELATIONS = frozenset(
    {
        "ca",
        "pz",
        "lda",
        "pb",
        "pbe",
        # PARSEC 1.5's wb is this port's PBE (_CORRELATION_LABELS), which both sets of routines run
        "wb",
    }
)
# The meta-GGAs of this port (SCF/meta_gga.py, SCF/meta_gga_periodic.py) run on either set of routines.
_DEFAULT_CORRELATIONS = _DEFAULT_CORRELATIONS | frozenset(_META_GGA_LABELS)
_DEFAULT_EIGENSOLVERS = frozenset({"chebff", "chebdav"})
_DEFAULT_FIELD_UNITS = frozenset({"", "ry/bohr/e", "rydberg/bohr/e", "ha/bohr/e", "hartree/bohr/e", "ev/ang/e"})


def _default_routines_value_refusal(scalar, one, optional_bool) -> str | None:
    """The first value of the input that Andrew Grieder's routines refuse, as a reason, or ``None``.

    Only labels his parser reads are looked at (an unknown label is decided by
    :func:`parsec_python.Input.routines.decide_routines`); the geometry, grid and combination conditions are
    checked where the parser has them.
    """

    if "restart_run" in scalar and optional_bool("restart_run"):
        return "Restart_Run: .true."
    if "old_pseudopotential_format" in scalar and optional_bool("old_pseudopotential_format"):
        return "Old_Pseudopotential_Format: .true."
    if "dynamic_diag_tol" in scalar and optional_bool("dynamic_diag_tol"):
        return "Dynamic_Diag_Tol: .true."
    if "old_interpolation_format" in scalar and optional_bool("old_interpolation_format"):
        return "Old_Interpolation_Format: .true."
    if "double_grid_order" in scalar:
        try:
            order = int(float(one("double_grid_order")))
        except ValueError:
            order = 1
        if order > 1:
            return "Double_Grid_Order > 1"
    if "electric_field" in scalar:
        text = one("electric_field")
        value = _first_float(text, label="Electric_Field")
        remainder = _FLOAT_PATTERN.sub("", text, count=1).strip().lower()
        unit = re.sub(r"[\s_-]", "", remainder)
        if value != 0.0 or unit not in _DEFAULT_FIELD_UNITS:
            return "a nonzero Electric_Field (or its unit)"
    if "spin_polarization" in scalar and optional_bool("spin_polarization") and "correlation_type" not in scalar:
        return "Spin_Polarization without Correlation_Type (PARSEC's default PW LSDA)"
    correlation = _normalize_label(one("correlation_type", "ca"))
    # A hybrid functional runs the Fock loop of SCF/hybrid.py and SCF/hybrid_periodic.py on either set of routines.
    if correlation not in _DEFAULT_CORRELATIONS and _hybrid_label(one("correlation_type", "ca")) is None:
        return f"Correlation_Type: {correlation}"
    shape = _normalize_label(one("cluster_domain_shape", "sphere"))
    if shape not in {"sphere", "box"}:
        return f"Cluster_Domain_Shape: {shape}"
    for item in scalar.get("pseudopotential_format", []):
        name = _normalize_label(str(item.value))
        if name not in {"martins_new", "martinsnew", "martins_old"}:
            return f"Pseudopotential_Format: {name}"
    eigensolver = _normalize_label(one("eigensolver", "chebdav"))
    if eigensolver not in _DEFAULT_EIGENSOLVERS:
        return f"Eigensolver: {eigensolver}"
    mixing = _normalize_label(one("mixing_method", "anderson"))
    if mixing != "anderson":
        return f"Mixing_Method: {mixing}"
    return None

def parse_parsec_input(
    path: str | Path,
    *,
    pseudopotential_directory: str | Path | None = None,
    esdf_lenient: bool = False,
) -> ParsecInputTranslation:
    """Translate a supported ``parsec.in`` (isolated or periodic) into ``SinglePointInput``.

    The routines of the run are decided here (:mod:`parsec_python.Input.routines`): an input that the default
    routines do not compute is translated again for the Fortran-derived ones.

    ``esdf_lenient`` (the CLIs' ``--esdf-lenient``) reads the input as PARSEC's ESDF does: a label this reader
    does not know is ignored and a label given twice takes its first value, each with a warning.  By default both
    are refused (an unknown label may be a misspelt keyword); the inputs of PARSEC 1.5's test suite hold such
    lines (``boundary_mask``, ``scfso``, ``nscf_kpoints (block)``, ``max_iter`` twice, ...).
    """
    try:
        try:
            return _parse_parsec_input(
                path,
                pseudopotential_directory=pseudopotential_directory,
                esdf_lenient=esdf_lenient,
            )
        except UseFortranRoutines as fallback:
            return _parse_parsec_input(
                path,
                pseudopotential_directory=pseudopotential_directory,
                routines=fallback.choice,
                esdf_lenient=esdf_lenient,
            )
    except ParsecInputError:
        raise
    except ValueError as error:
        raise ParsecInputError(str(error)) from error


def _mixing_summary(mixing: MixingSettings) -> str:
    """The mixer line of :func:`summarize_translation`."""

    if mixing.method == "anderson":
        return (
            f"Mixing: Anderson alpha={mixing.parameter:.6g}, "
            f"memory={mixing.memory}, restart={mixing.restart}, "
            f"safeguard={mixing.safeguard}"
        )
    if mixing.method == "broyden":
        return (
            f"Mixing: generalized Broyden alpha={mixing.parameter:.6g}, "
            f"memory={mixing.memory}, restart={mixing.restart}"
        )
    return (
        f"Mixing: {mixing.method} alpha={mixing.parameter:.6g}, "
        f"memory={mixing.memory}, group_size={mixing.group_size}, "
        f"en_like={mixing.en_like}, restart_factor={mixing.restart_factor:.6g}, "
        f"expand_factor={mixing.expand_factor:.6g}"
        + (
            f", preferred_type={mixing.preferred_type}"
            if mixing.method == "msecant3"
            else ""
        )
    )


def _periodic_potential_conventions(
    scalar, one, optional_bool, grid, scf, *, is_periodic, wire_or_slab, warnings, bulk=None
):
    """``Periodic_Local_Potential``, ``Periodic_Alpha_Term``, ``Zero_Average_Vxc``.

    Python extensions (Fortran has no keywords: it always does the second
    choice) for a 3-D periodic cell:

    * ``Periodic_Local_Potential: real_space`` (default; this port's Ewald
      split) or ``g_space`` (Fortran's ``v_first``/``pot_local``);
    * ``Periodic_Alpha_Term: analytic`` (default, the ``q -> 0`` limit) or
      ``fortran`` (the finite difference of ``fourier_f.f90``), with
      ``g_space`` only;
    * ``Zero_Average_Vxc``: Fortran's ``potshift``, the average of the
      exchange-correlation potential of each spin channel removed at every
      evaluation.  The default since 2026-11-08 for an unpolarized collinear
      3-D cell (PARSEC 1.5 always does it there; the total energy, the density
      and the forces do not change, the eigenvalues and the Fermi level lose
      the average of V_xc); ``.false.`` restores the earlier potential.  A
      spin-polarized cell keeps it off unless asked for: each channel losing
      its own average moves the channels against each other, which changes the
      occupations and the energy of a magnetic cell.

    A cluster reads none of them (a warning says so); a wire or slab refuses
    the G-space potential (Fortran's ``ionpot_wire``/``ionpot_slab`` build
    theirs otherwise) and ignores ``Zero_Average_Vxc`` as Fortran does
    (``pbc%per == 3`` only, with a warning); non-collinear magnetism refuses
    ``Zero_Average_Vxc``.
    """

    given = [
        key
        for key in ("periodic_local_potential", "periodic_alpha_term", "zero_average_vxc")
        if key in scalar
    ]
    if (
        "zero_average_vxc" not in scalar
        and is_periodic
        and not wire_or_slab
        and (bulk is None or bulk)
        and not scf.spin_polarized
        and not scf.noncollinear
    ):
        # PARSEC 1.5's potshift of a 3-D cell (not a slab of either routines), by default
        scf = replace(scf, zero_average_xc_potential=True)
    if not given:
        return grid, scf
    if not is_periodic:
        warnings.append(
            "Periodic_Local_Potential, Periodic_Alpha_Term and Zero_Average_Vxc are "
            "read for a periodic cell only; no effect here"
        )
        return grid, scf
    local = _normalize_label(one("periodic_local_potential", "real_space"))
    local = {"realspace": "real_space", "gspace": "g_space"}.get(local, local)
    if local not in ("real_space", "g_space"):
        raise ParsecInputError(
            "Periodic_Local_Potential must be real_space (this port's Ewald split) "
            "or g_space (Fortran's v_first/pot_local)"
        )
    alpha = _normalize_label(one("periodic_alpha_term", "analytic"))
    alpha = {"finite_difference": "finite_difference", "finitedifference": "finite_difference", "fortran": "finite_difference"}.get(alpha, alpha)
    if alpha not in ("analytic", "finite_difference"):
        raise ParsecInputError(
            "Periodic_Alpha_Term must be analytic or fortran (finite_difference)"
        )
    if wire_or_slab and local == "g_space":
        raise UnsupportedParsecOptionError(
            "Periodic_Local_Potential: g_space is implemented for a 3-D periodic "
            "cell; a wire or slab has its own local potential (Fortran "
            "ionpot_wire/ionpot_slab)"
        )
    if alpha != "analytic" and local != "g_space":
        raise UnsupportedParsecOptionError(
            "Periodic_Alpha_Term belongs to Periodic_Local_Potential: g_space (the "
            "real-space Ewald split carries the average of the potential itself)"
        )
    if local != "real_space" or alpha != "analytic":
        grid = replace(grid, local_potential=local, alpha_term=alpha)
    if optional_bool("zero_average_vxc"):
        if wire_or_slab:
            warnings.append(
                "Zero_Average_Vxc: PARSEC's potshift is applied to 3-D periodic cells "
                "only (rho_hart.F90, pbc%per == 3); no effect for a wire or slab"
            )
        elif scf.noncollinear:
            raise UnsupportedParsecOptionError(
                "Zero_Average_Vxc (PARSEC's potshift) is not implemented for "
                "non-collinear magnetism"
            )
        else:
            scf = replace(scf, zero_average_xc_potential=True)
            if scf.spin_polarized:
                warnings.append(
                    "Zero_Average_Vxc with spin polarization: as PARSEC's potshift, each "
                    "channel loses its own average, which also removes the average "
                    "exchange splitting between the channels (one Fermi level for both)"
                )
    return grid, scf


def summarize_translation(translation: ParsecInputTranslation) -> str:
    """Return a concise, deterministic input summary for CLI/log output."""
    problem = translation.problem
    if problem.eigensolver.method == "chebff":
        first_solver_summary = (
            f"first_filter={problem.eigensolver.first_filter_degree}"
            f"x{problem.eigensolver.first_filter_cycles}"
        )
    elif problem.eigensolver.method == "chebdav":
        first_solver_summary = (
            f"chebdav_degree={problem.eigensolver.first_filter_degree}"
        )
    elif problem.eigensolver.method == "none":
        first_solver_summary = "TEST mode (setup only)"
    else:
        first_solver_summary = (
            f"first_solver={problem.eigensolver.method}, "
            "chebyshev_filtering="
            f"{problem.eigensolver.filters_later_iterations}, "
            f"maximum_matvec={problem.eigensolver.maximum_matvec}"
        )
    if problem.eigensolver.dynamic_tolerance:
        first_solver_summary += ", dynamic_diag_tol=True"
    if problem.periodic_cell is None:
        grid_summary = (
            f"Grid: {problem.grid.domain_shape}, h={problem.grid.spacing:.12g} bohr, "
            f"order={problem.grid.expansion_order}, shift={problem.grid.shift}"
        )
    elif getattr(problem.grid, "periodicity", 3) in (1, 2):
        # A wire (cylinder) or slab: the periodic lattice vectors and radius.
        vectors = problem.periodic_cell.lattice_vectors[: problem.grid.periodicity]
        grid_summary = (
            f"Grid: {'wire' if problem.grid.periodicity == 1 else 'slab'}, "
            "lattice vectors="
            + str([[float(f"{value:.10g}") for value in row] for row in vectors])
            + f" bohr, Boundary_Sphere_Radius={problem.grid.confinement_radius:.10g} bohr, "
            f"h_target={problem.grid.spacing:.12g} bohr, "
            f"order={problem.grid.expansion_order}, shift={problem.grid.shift}"
        )
        if problem.charged_sheets:
            grid_summary += f", charged sheets={len(problem.charged_sheets)}"
    elif getattr(problem.grid, "fortran_routines", False) and uses_general_periodic_grid(
        problem.periodic_cell, problem.grid
    ):
        # Fortran's grid of a general cell: the lattice vectors (rows).
        vectors = problem.periodic_cell.lattice_vectors
        grid_summary = (
            "Grid: periodic (general lattice), lattice vectors="
            + str([[float(f"{value:.10g}") for value in row] for row in vectors])
            + f" bohr, h_target={problem.grid.spacing:.12g} bohr, "
            f"order={problem.grid.expansion_order}, shift={problem.grid.shift}"
        )
    else:
        vectors = problem.periodic_cell.lattice_vectors
        side_lengths = (
            np.diag(vectors)
            if _lattice_is_diagonal(vectors)
            else np.linalg.norm(vectors, axis=1)
        )
        grid_summary = (
            f"Grid: periodic, cell={list(side_lengths)} bohr, "
            f"h_target={problem.grid.spacing:.12g} bohr, "
            f"order={problem.grid.expansion_order}"
        )
    if problem.grid.double_grid_order > 1:
        grid_summary += f", double_grid_order={problem.grid.double_grid_order}"
    if any(item.old_interpolation for item in problem.pseudopotentials.values()):
        grid_summary += ", old (direct) interpolation of the local pseudopotential"
    lines = [
        f"Input: {translation.source}",
        f"Atoms: {len(problem.atoms)}",
        f"Species: {', '.join(problem.pseudopotentials)}",
        grid_summary,
        #AG (
        #AG     f"Grid: {problem.grid.domain_shape}, h={problem.grid.spacing:.12g} bohr, "
        #AG     f"order={problem.grid.expansion_order}, shift={problem.grid.shift}"
        #AG ),
        (
            f"SCF: states={problem.scf.number_of_states}, "
            f"max_iter={problem.scf.max_iterations}, "
            f"vconv={problem.scf.convergence_criterion:.6g} Ry, "
            + (
                f"T={problem.scf.fermi_temperature_kelvin:.6g} K"
                if problem.scf.fixed_occupations is None
                else "occupations fixed by occup.in"
            )
        ),
        (
            f"Eigensolver: {problem.eigensolver.method}, "
            f"tol={problem.eigensolver.tolerance:.3g}, "
            f"{first_solver_summary}, "
            f"block={problem.eigensolver.matvec_block_size}, "
            f"later_filter={problem.eigensolver.filter_degree}"
            f"+/-{problem.eigensolver.filter_degree_delta}, "
            f"buffer={problem.eigensolver.subspace_buffer}"
        ),
        _mixing_summary(problem.mixing)
        + (
            ""
            if problem.mixing.preconditioner == "none"
            else (
                f", Kerker preconditioner q0={problem.mixing.kerker_wavevector:.4g} bohr^-1"
                + (
                    ""
                    if problem.mixing.kerker_dielectric_constant == float("inf")
                    else f" eps={problem.mixing.kerker_dielectric_constant:.4g}"
                )
            )
        ),
        (
            "Initial density: "
            f"{problem.initial_density_settings.method}, "
            f"normalize={problem.scf.normalize_initial_density}"
        ),
        *(
            ()
            if translation.mlff is None
            else (
                f"Initial structure: relaxed with the force field {translation.mlff.model}"
                + (
                    f" (task {translation.mlff.task})"
                    if translation.mlff.model.startswith(("uma", "esen", "allscaip"))
                    else ""
                )
                + f", fmax={translation.mlff.fmax:g} eV/ang, steps={translation.mlff.steps}, "
                f"{translation.mlff.optimizer}, device={translation.mlff.device}",
            )
        ),
        # A periodic cell has no Hartree boundary values to control.
        *(
            ()
            if problem.periodic_cell is not None
            else (
                f"Hartree boundary: Solver_Lpole={problem.hartree.multipole_order}, "
                "tolerance="
                + (
                    "off"
                    if problem.hartree.boundary_tolerance is None
                    else f"{problem.hartree.boundary_tolerance:.6g} Ry"
                )
                + f", atomic_tail={problem.hartree.atomic_tail}"
                + (
                    " (direct Coulomb sum)"
                    if problem.hartree.boundary_method == "direct"
                    else ""
                ),
            )
        ),
        # PARSEC 1.5's Poisson CG (Hartree_CG_Tolerance / Hartree_CG_Criterion): only where it is asked for.
        *(
            (
                f"Hartree CG: PARSEC 1.5's, relative tolerance {problem.hartree.relative_tolerance:.1e} of the "
                + ("right-hand side" if getattr(problem.hartree, "cg_criterion", 1) == 2 else "initial residual")
                + f", at most {problem.hartree.max_iterations} products",
            )
            if (problem.hartree.relative_tolerance, problem.hartree.max_iterations, getattr(problem.hartree, "cg_criterion", 1))
            != (1.0e-7, 1600, 1)
            else ()
        ),
        (
            f"Output: level={translation.output_level}, "
            f"all_states={translation.output_all_states}"
        ),
        # Only for a spin-polarized run: other summaries keep their text.
        *(
            (
                "Spin: polarized, initial polarization "
                + ", ".join(
                    f"{symbol} {potential.initial_spin_polarization:g}"
                    for symbol, potential in problem.pseudopotentials.items()
                ),
            )
            if problem.scf.spin_polarized
            else ()
        ),
        # Only where they are asked for: other summaries keep their text.
        *(
            ()
            if translation.polarizability_field is None
            else (
                "Polarizability: seven-field approach, Electric_Field="
                f"{translation.polarizability_field:.6g} ry/bohr/e",
            )
        ),
        *(
            ()
            if not problem.point_charges
            else (
                f"Point charges: {len(problem.point_charges)}, total "
                f"{sum(item.charge for item in problem.point_charges):.6g} e",
            )
        ),
        *(
            (
                "Charge density: symmetrized over the point group"
                if problem.periodic_cell is None
                else "Charge density: symmetrized over the space group",
            )
            if problem.scf.symmetrize_density
            else ()
        ),
        # Only for Fortran's k-point path (a Monkhorst_Pack_Shift block).
        *(
            ()
            if getattr(problem, "monkhorst_pack_shift", None) is None
            else (
                "K-points: Monkhorst-Pack "
                + "x".join(str(value) for value in problem.monkhorst_pack_dimensions)
                + ", shift "
                + " ".join(f"{value:g}" for value in problem.monkhorst_pack_shift)
                + " (PARSEC's grid, reduced to the irreducible wedge by "
                + (
                    "the identity"
                    if problem.space_group is None or problem.space_group.order == 1
                    else f"{problem.space_group.order} space-group operations"
                )
                + (" without time reversal)" if problem.scf.noncollinear else " and time reversal)"),
            )
        ),
        # Only where the refinement is requested: other summaries keep their text.
        *(
            ()
            if translation.energy_refinement is None
            else (
                f"Energy refinement: {translation.energy_refinement} "
                "(one post-SCF evaluation on the converged orbitals)",
            )
        ),
        # Lines of the post-processing keywords only where the input has them.
        *(
            (
                f"NSCF: {len(translation.nscf.weights)} k-points, "
                f"nscf_states_num={translation.nscf.number_of_states}, potential of "
                "parsec.dat (no SCF)",
            )
            if getattr(translation, "nscf", None) is not None
            else ()
        ),
        *(
            (
                f"DOS: DOS_Pnum={translation.dos.points}"
                + (
                    ""
                    if translation.dos.monkhorst_pack_grid is None
                    else f", grid={translation.dos.monkhorst_pack_grid}"
                )
                + (
                    f", YLM_DOS_L={translation.dos.ylm_l}"
                    if translation.dos.ylm_l >= 0
                    else ""
                ),
            )
            if getattr(translation, "dos", None) is not None
            else ()
        ),
        *(
            (
                f"Output_GW: BerkeleyGW files WFN, RHO, VXC; k-grid {translation.gw.kgrid}, "
                f"shift {translation.gw.kshift}, {translation.gw.space_group.order} symmetry "
                "operations",
            )
            if getattr(translation, "gw", None) is not None
            else ()
        ),
        *(
            ("Hartree: Calc_Hartree_G (V_H(G) = 8 pi rho(G)/|G|^2 by FFT)",)
            if getattr(problem.hartree, "periodic_g_space", False)
            else ()
        ),
        *(
            (
                "Local potential: Fortran's G-space construction (v_first/pot_local), "
                f"alpha term {problem.grid.alpha_term.replace('_', ' ')}",
            )
            if getattr(problem.grid, "local_potential", "real_space") == "g_space"
            else ()
        ),
        *(
            ("Exchange-correlation potential: average removed every iteration (potshift)",)
            if getattr(problem.scf, "zero_average_xc_potential", False)
            else ()
        ),
    ]
    for symbol, potential in problem.pseudopotentials.items():
        lines.append(
            f"PP {symbol}: {potential.path} "
            f"(local l={potential.local_angular_momentum}, "
            f"Read_VCD={potential.read_valence_density}"
            + (
                ""
                if potential.pseudopotential_format == "martins_new"
                else (
                    f", {potential.pseudopotential_format}: "
                    f"Core_Cutoff_Radius={potential.core_cutoff_radius:.6g} bohr"
                )
                if potential.pseudopotential_format == "martins_wang"
                else (
                    f", {potential.pseudopotential_format}: "
                    f"Potential_Num={potential.number_of_channels}, "
                    f"Electron_Per_Orbital={potential.electrons_per_orbital}, "
                    f"Core_Cutoff_Radius={potential.core_cutoff_radius:.6g} bohr"
                    + (
                        ", Nonlinear_Core_Correction=True"
                        if potential.nonlinear_core_correction
                        else ""
                    )
                )
            )
            + (
                f", Alpha_Filter={potential.fourier_filter_alpha:g}, "
                f"Beta1_Filter={potential.fourier_filter_beta1:g}, Core_Filter="
                + (
                    f"{potential.fourier_filter_core:g}"
                    if potential.fourier_filter_core is not None
                    else f"{potential.fourier_filter_alpha:g}"
                )
                if potential.has_fourier_filter
                or potential.fourier_filter_beta1 != 0.0
                or potential.fourier_filter_core is not None
                else ""
            )
            + ")"
        )
    return "\n".join(lines)


__all__ = [
    "ANGSTROM_TO_BOHR",
    "EV_TO_RYDBERG",
    "ParsecInputError",
    "ParsecInputTranslation",
    "UnsupportedParsecOptionError",
    "parse_parsec_input",
    "summarize_translation",
]
