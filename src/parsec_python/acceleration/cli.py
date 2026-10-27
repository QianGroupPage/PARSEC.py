"""Command-line runner for the additive accelerated single-point package."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import time
from typing import Sequence

from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.Input import (
    ParsecInputError,
    parse_parsec_input,
    summarize_translation,
)
from parsec_python.Output.parsec_output import domain_dry_run_lines
from parsec_python.V_ion import load_pseudopotentials
from parsec_python.MolecularDynamics.run import (
    restart_structure as dynamics_restart_structure,
    run_recorded_molecular_dynamics,
)
from parsec_python.Relaxation.run import (
    relaxation_header_lines,
    restart_structure,
    run_recorded_relaxation,
    run_single_point_with_forces,
    symmetry_decision,
)
from parsec_python.cli import (
    main as reference_main,
    refinement_progress,
    relaxation_exit_code,
    save_result_archive,
    save_self_consistent_soc_result_archive,
)

from .Output import AcceleratedTextReporter
from .backends.selection import resolve_backend
from .SCF.single_point import materialize_wavefunctions
from .driver import (
    _cupy_installed,
    hartree_boundary_check_seconds,
    ionic_backend,
    prepare_single_point,
    profile_hamiltonian_components,
    run_scf,
)
from .models import BackendUnavailableError
from .SCF.self_consistent_soc import run_self_consistent_soc
from .resources import Resources, native_thread_information, resolve_resources


class _RunLog:
    def __init__(self, path: Path | None, quiet: bool = False) -> None:
        self.path = path
        self.quiet = quiet
        self._stream = None
        self.failure_reported = False

    def __enter__(self) -> "_RunLog":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, _type, value, _traceback) -> None:
        if value is not None:
            message = (
                "Calculation interrupted by user."
                if isinstance(value, KeyboardInterrupt)
                else f"Calculation failed: {type(value).__name__}: {value}"
            )
            try:
                self.write(message)
                self.failure_reported = True
            except OSError:
                pass
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def write(self, message: str = "") -> None:
        if not self.quiet:
            print(message, flush=True)
        if self._stream is not None:
            self._stream.write(message + "\n")
            self._stream.flush()


def _resolve_output(
    requested: Path | None,
    input_directory: Path,
    default_name: str,
) -> Path:
    if requested is None:
        return input_directory / default_name
    requested = requested.expanduser()
    if requested.is_absolute():
        return requested.resolve()
    return (input_directory / requested).resolve()


def _npz_path(path: Path) -> Path:
    return path if path.suffix.lower() == ".npz" else path.with_suffix(path.suffix + ".npz")


def _symmetry_cache_directory(arguments: argparse.Namespace) -> Path | None:
    """Return the symmetry cache directory the command line named, if any.

    A first calculation of a structure is fastest without a cache, so none
    is read or written unless ``--symmetry-cache`` names its directory;
    ``--no-symmetry-cache`` spells that default out.  A relative directory
    is taken from the working directory, as it always was.
    """

    if arguments.symmetry_cache is None:
        return None
    return arguments.symmetry_cache.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Run the modular isolated real-space DFT solver. By default it runs "
            "on the GPU where one is usable and on the CPU otherwise (--device "
            "auto), on every GPU the process can see and with the CPUs it may "
            "use less four; --device, --gpus and --threads name them explicitly. "
            "On the GPU the Hamiltonian and the eigensolver run in CuPy and, "
            "with the native extension, finite-difference construction and "
            "Hartree in C++; results are in double precision either way."
        ),
    )
    parser.add_argument("input", type=Path, help="PARSEC input file, usually parsec.in")
    parser.add_argument(
        "--restart-from",
        type=Path,
        default=None,
        help=(
            "take the atom coordinates from this file instead of the input: a "
            "relax_restart.dat (its BFGS state continues a BFGS relaxation) or "
            "an .xyz file (last frame); Relax_Restart in the input reads "
            "relax_restart.dat beside it"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default=None,
        help=(
            "where the calculation runs: auto (default) takes the GPU if CuPy "
            "finds a usable one and the CPU otherwise; gpu requires a GPU and "
            "stops where there is none; cpu never touches a GPU. The variable "
            "PARSEC_DEVICE gives the same choice to a job script; parsec.out "
            "says what ran"
        ),
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpus",
        default=None,
        metavar="N|all",
        help=(
            "how many GPUs of this node to use: all (default) is every GPU the "
            "process can see, which in a scheduler job is what the job was "
            "given; N takes the first N of them. Naming a number asks for a "
            "GPU run"
        ),
    )
    gpu_group.add_argument(
        "--gpu-ids",
        default=None,
        metavar="LIST",
        help=(
            "the GPUs to use by index, such as 0,2: indices count the GPUs the "
            "process can see (inside CUDA_VISIBLE_DEVICES, which is left alone)"
        ),
    )
    parser.add_argument(
        "--threads",
        default=None,
        metavar="N|all|auto",
        help=(
            "CPU threads of the C++/OpenMP kernels: auto (default) follows "
            "OMP_NUM_THREADS and otherwise takes the CPUs the process may use "
            "less four, all takes every one of them, N names the number. The "
            "CPUs are those of the allocation in a scheduler job, not those of "
            "the machine"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "scipy", "native", "cupy"),
        default="auto",
        help=(
            "which implementation executes, for controlled comparisons; "
            "--device is the usual choice. auto (default) follows --device: on "
            "the GPU it combines native setup/Hartree with the CuPy "
            "Hamiltonian/eigensolver, on the CPU it takes native C++ or, "
            "without the extension, SciPy. Explicit scipy and native (CPU) and "
            "cupy (GPU) never fall back; every selected component and fallback "
            "is recorded"
        ),
    )
    parser.add_argument(
        "--symmetry",
        choices=("auto", "on", "off"),
        default=None,
        help=(
            "symmetry policy: auto (default) detects exact supported "
            "operations and falls back safely, on requires a nontrivial "
            "usable symmetry, and off forces full-grid calculations; "
            "Ignore_Symmetry=true selects off unless this option is given"
        ),
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--symmetry-cache",
        type=Path,
        default=None,
        metavar="DIRECTORY",
        help=(
            "read and write exact-key symmetry maps, representation "
            "operators and Hartree boundary geometry in DIRECTORY; it pays "
            "only for repeated calculations of the same structure and grid, "
            "and no directory is read or written unless it is named here"
        ),
    )
    cache_group.add_argument(
        "--no-symmetry-cache",
        action="store_true",
        help="use no persistent symmetry cache (the default)",
    )
    parser.add_argument(
        "--pp-dir",
        type=Path,
        default=None,
        help="additional pseudopotential directory searched before the input directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="result .npz path; defaults beside parsec.in",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="text log path; defaults to parsec.out beside parsec.in",
    )
    parser.add_argument(
        "--kpoint-workers",
        default=None,
        metavar="N|auto",
        help=(
            "worker threads for the k-points of a periodic SCF and for the band path "
            "(reference solver; default PARSEC_KPOINT_WORKERS, else auto)"
        ),
    )
    parser.add_argument(
        "--bands-from",
        type=Path,
        default=None,
        metavar="ARCHIVE.npz",
        help=(
            "skip the SCF and compute the bandstruc bands from the converged potential in a "
            "previous run's result archive (reference solver; log: parsec_bands.out)"
        ),
    )
    parser.add_argument(
        "--mpi",
        action="store_true",
        help=(
            "run as one of several MPI ranks (start them with mpirun or srun, one per GPU): "
            "the basis of the k-point and non-collinear eigenproblems is split by blocks of grid "
            "rows over all the GPUs, so a basis larger than one GPU fits. Every rank runs the same "
            "calculation; only rank 0 writes the log and the archive. Same as PARSEC_MPI=1"
        ),
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="do not write parsec_python_results.npz",
    )
    parser.add_argument(
        "--save-wavefunctions",
        action="store_true",
        help="include wavefunctions in the result archive",
    )
    parser.add_argument(
        "--mlff",
        nargs="?",
        const="",
        default=None,
        metavar="MODEL",
        help=(
            "relax the input structure with a machine-learned force field first "
            "(Initial_Structure: mlff); MODEL as MLFF_Model, for example uma-s-1p2, "
            "mace-mp-medium-mpa-0 or a file.model; without MODEL the input's or uma-s-1p2"
        ),
    )
    parser.add_argument(
        "--energy-refinement",
        choices=("b3lyp", "none"),
        default=None,
        help=(
            "after the SCF, evaluate the exchange-correlation energy of this "
            "hybrid functional once on the converged orbitals and report the "
            "refined total energy (PP-PBE(B3LYP) with a PBE input; ARES "
            "Lrefine); overrules Energy_Refinement of the input, none "
            "switches it off"
        ),
    )
    parser.add_argument(
        "--profile-operator",
        action="store_true",
        help=(
            "run an opt-in synchronized microprofile of finite-difference, "
            "local ionic, and nonlocal ionic Hamiltonian actions"
        ),
    )
    parser.add_argument(
        "--profile-repeats",
        type=int,
        default=3,
        help="component-profile repeats averaged when --profile-operator is used",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse/validate input and resolve backend without constructing the grid",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress console progress while retaining the text log",
    )
    parser.add_argument(
        "--resident",
        action="store_true",
        help=(
            "submit to the local warmed worker, auto-starting it if needed; "
            "handled by the folder-local launcher before ordinary parsing"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="re-raise runtime exceptions with a traceback",
    )
    parser.epilog = (
        "Resident controls: main.py --resident-start, --resident-status, "
        "or --resident-stop. Each request creates fresh DFT/SCF state while "
        "reusing only process and accelerator runtime state."
    )
    return parser


def _visible_gpus(resources: Resources) -> int:
    """Count the GPUs of this process and hold a named number or list against them."""

    from .backends.cupy import CuPyUnavailableError, cupy_device_count

    try:
        visible = cupy_device_count()
    except CuPyUnavailableError as error:
        raise BackendUnavailableError(
            f"a GPU run was requested ({resources.device_source}) but no GPU is usable: {error}"
        ) from error
    resources.check_gpus(visible)
    return visible


def _resource_report(resources: Resources, system) -> list[str]:
    """The lines of ``parsec.out`` that say what the calculation runs on."""

    info = system.backend_info
    gpus = visible = None
    if info.selected == "cupy":
        from .Eigensolvers.symmetry import selected_device_ids
        from .backends.cupy import cupy_device_count, require_cupy

        cp, _ = require_cupy()
        visible = cupy_device_count()
        gpus = selected_device_ids(int(cp.cuda.Device().id))
    return resources.report_lines(
        info.selected,
        fallback_reasons=tuple(info.fallback_reasons),
        gpus=gpus,
        visible_gpus=visible,
        native=native_thread_information(),
    )



def _accelerated_self_consistent_soc_eligible(translation) -> bool:
    """Whether this input matches the one self-consistent-SOC shape the
    accelerated path implements: Gamma-point only, not combined with
    collinear spin polarization.

    See :mod:`parsec_python.acceleration.SCF.self_consistent_soc`'s module
    docstring.  Combined with spin polarization or a periodic
    ``Boundary_Conditions`` always delegates to the reference CPU solver
    (:mod:`parsec_python.SCF.self_consistent_soc_spin_polarized`/
    ``kpoints_soc`` have no CuPy counterpart yet).
    """

    problem = translation.problem
    return (
        problem.scf.self_consistent_spin_orbit
        and not problem.scf.spin_polarized
        and not problem.scf.noncollinear
        and problem.periodic_cell is None
    )


def _resolve_symmetry_mode(
    requested_symmetry: str | None,
    ignore_symmetry: bool,
    *,
    accelerated_self_consistent_soc: bool,
) -> str:
    """Resolve ``--symmetry``/``Ignore_Symmetry`` to an effective mode.

    Self-consistent SOC on the accelerated path always forces "off": the
    orbital-sector symmetry decomposition
    (``CuPySymmetrySCFEigensolver``/``SymmetrySCFReducer``) assumes
    real-valued, spin-diagonal 1-D irreps, but the spin-orbit L.S cross term
    (``lzsz``/``lsxy``) mixes spin channels and breaks that assumption, and
    ``CuPySpinorHamiltonian``/``run_chebff_spinor`` only accept plain
    complex128 ndarrays, not the compact ``SymmetryScalarField``
    representation the symmetry-reduced Hartree/XC/mixing hooks would
    otherwise produce for this system. "auto"/unset falls back to full-grid
    silently (its own documented contract); "on" explicitly promises a
    nontrivial reduction, which this tier cannot deliver, so that
    combination is rejected with a ``ValueError`` instead of silently
    overriding the user's explicit request.
    """

    mode = (
        requested_symmetry
        if requested_symmetry is not None
        else ("off" if ignore_symmetry else "auto")
    )
    if accelerated_self_consistent_soc:
        if requested_symmetry == "on":
            raise ValueError(
                "self-consistent spin-orbit coupling on the accelerated "
                "path does not yet support symmetry reduction, so "
                "--symmetry on cannot be satisfied for this input; omit "
                "--symmetry or pass --symmetry off"
            )
        return "off"
    return mode


def _reference_only_features(
    translation, *, accelerated_backend_selected: bool
) -> list[str]:
    """Input features only the reference :mod:`parsec_python.cli` dispatches.

    The accelerated driver runs a scalar Gamma-point SCF, spin-polarized
    where the system is isolated and carries no spin-orbit species, plus (when ``accelerated_backend_selected`` and the input is
    Gamma-point-only self-consistent SOC without spin polarization --
    :func:`_accelerated_self_consistent_soc_eligible`) the CuPy spinor
    Hamiltonian/eigensolver in
    :mod:`parsec_python.acceleration.SCF.self_consistent_soc`.  Without this
    check every other SOC/spin/k-point flag would be silently ignored.
    """

    problem = translation.problem
    features = []
    if problem.scf.noncollinear:
        features.append("non-collinear magnetism (Non_Collinear_magnetism)")
    elif problem.scf.self_consistent_spin_orbit:
        if problem.scf.spin_polarized:
            features.append(
                "self-consistent spin-orbit coupling combined with spin "
                "polarization (SO_from_scratch + Spin_Polarization)"
            )
        elif problem.periodic_cell is not None:
            features.append(
                "periodic self-consistent spin-orbit coupling "
                "(SO_from_scratch with a periodic Boundary_Conditions)"
            )
        elif not accelerated_backend_selected:
            features.append("self-consistent spin-orbit coupling (SO_from_scratch)")
        # else: Gamma-point-only self-consistent SOC on a resolved CuPy
        # backend -- accelerated, not reference-only.
    elif problem.scf.spin_polarized and (
        problem.periodic_cell is not None
        or any(
            specification.spin_orbit
            for specification in problem.pseudopotentials.values()
        )
    ):
        features.append(
            "spin polarization of a periodic system or of spin-orbit species"
        )
    if problem.has_kpoint_sampling and not (
        (problem.scf.self_consistent_spin_orbit or problem.scf.noncollinear)
        and problem.periodic_cell is not None
    ):
        features.append("k-point sampling (Monkhorst-Pack or explicit list)")
    if problem.bands is not None:
        features.append("band structure along a k-path (bandstruc)")
    return features


def _reference_argv(arguments: argparse.Namespace) -> list[str]:
    """Forward the options the reference CLI shares with this one."""

    argv = [str(arguments.input)]
    for flag, value in (
        ("--pp-dir", arguments.pp_dir),
        ("--output", arguments.output),
        ("--log", arguments.log),
        ("--bands-from", arguments.bands_from),
        ("--kpoint-workers", arguments.kpoint_workers),
        ("--restart-from", arguments.restart_from),
        ("--energy-refinement", arguments.energy_refinement),
    ):
        if value is not None:
            argv += [flag, str(value)]
    if arguments.mlff is not None:
        argv += ["--mlff"] + ([arguments.mlff] if arguments.mlff else [])
    for flag, enabled in (
        ("--no-archive", arguments.no_archive),
        ("--save-wavefunctions", arguments.save_wavefunctions),
        ("--dry-run", arguments.dry_run),
        ("--quiet", arguments.quiet),
        ("--debug", arguments.debug),
    ):
        if enabled:
            argv.append(flag)
    return argv


def _join_mpi(arguments: argparse.Namespace) -> int | None:
    """Make this process one rank of an MPI run; ``None`` if all is well, else an exit status.

    Every rank runs the whole calculation (see :mod:`.Parallel.mpi`); the ranks other than 0 are
    silent and write no files.
    """
    from .Parallel import mpi

    if not (arguments.mpi or mpi.requested()):
        return None
    os.environ[mpi.MPI_ENV] = "1"
    try:
        context = mpi.context()
    except ImportError as error:
        print(f"Input error: --mpi needs mpi4py: {error}", file=sys.stderr)
        return 2
    if arguments.dry_run:
        return None
    if not context.is_root:
        arguments.quiet = True
        arguments.log = Path(os.devnull)
        arguments.no_archive = True
        arguments.output = None
    elif context.size > 1:
        print(
            f"MPI: {context.size} ranks, {context.ranks_per_node} per node; the eigenproblem basis "
            "is split by blocks of grid rows over the ranks",
            flush=True,
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    status = _join_mpi(arguments)
    if status is not None:
        return status
    try:
        resources = resolve_resources(
            device=arguments.device,
            gpus=arguments.gpus,
            gpu_ids=arguments.gpu_ids,
            threads=arguments.threads,
            backend=arguments.backend,
        )
    except ValueError as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    # The request reaches the solver through its variables, for this
    # calculation only: a process that runs several keeps none of them.
    with resources.applied():
        try:
            status = _run(arguments, resources)
        except BaseException:
            _abort_mpi(1)
            raise
    if status not in (0, 3):
        _abort_mpi(status)
    return status


def _abort_mpi(status: int) -> None:
    """A rank that failed must not leave the others waiting in a collective (the interpreter's exit
    would wait for them in MPI_Finalize): end the whole MPI job."""
    from .Parallel import mpi

    context = mpi.context()
    if context is not None and context.size > 1:
        sys.stdout.flush()
        sys.stderr.flush()
        context.comm.Abort(int(status))


def _run(arguments: argparse.Namespace, resources: Resources) -> int:
    if arguments.profile_repeats < 1:
        print("Input error: --profile-repeats must be positive", file=sys.stderr)
        return 2
    if arguments.bands_from is not None:
        # Bands from a stored SCF potential only exist in the reference solver.
        return reference_main(_reference_argv(arguments))
    try:
        translation = parse_parsec_input(
            arguments.input,
            pseudopotential_directory=arguments.pp_dir,
        )
    except (ParsecInputError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    if arguments.mlff is not None:
        from parsec_python.MLFF import apply_mlff_option

        translation = apply_mlff_option(translation, arguments.mlff)
    if arguments.energy_refinement is not None:
        from parsec_python.Energy.refinement import apply_refinement_option

        try:
            translation = apply_refinement_option(
                translation, arguments.energy_refinement
            )
        except ValueError as error:
            print(f"Input error: {error}", file=sys.stderr)
            return 2

    from parsec_python.MLFF import MLFFError, relax_initial_structure

    try:
        problem_of_input = translation.problem
        translation, relax_restart, restart_notes = restart_structure(
            translation,
            translation.source.parent,
            None
            if arguments.restart_from is None
            else _resolve_output(arguments.restart_from, translation.source.parent, ""),
        )
        translation, dynamics_restart, dynamics_notes = dynamics_restart_structure(
            translation, translation.source.parent
        )
        restart_notes = [*restart_notes, *dynamics_notes]
        # Initial_Structure: mlff, unless a restart file gave the structure;
        # a dry run only reports the settings.
        mlff_notes = []
        if not arguments.dry_run:
            translation, mlff_notes = relax_initial_structure(
                translation,
                translation.source.parent,
                restarted=translation.problem is not problem_of_input,
                progress=None if arguments.quiet else print,
            )
        restart_notes = [*restart_notes, *mlff_notes]
        use_symmetry, symmetry_note = symmetry_decision(translation)
        # Whether the atoms stay in the point group of the first structure,
        # which Ignore_Symmetry does not tell.  The command line may overrule
        # that keyword, but not a run that leaves the group.
        keeps_group, group_note = symmetry_decision(
            replace(translation, ignore_symmetry=False)
        )
    except MLFFError as error:
        print(f"MLFF error: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    if arguments.symmetry == "on" and not keeps_group:
        print(
            "Input error: --symmetry on cannot be used here: "
            + (
                group_note
                or "manual movements and molecular dynamics use no symmetry"
            ),
            file=sys.stderr,
        )
        return 2
    symmetry_mode = (
        arguments.symmetry
        if arguments.symmetry is not None
        and (keeps_group or arguments.symmetry == "off")
        else ("auto" if use_symmetry else "off")
    )
    # Forces are symmetrized where the calculation itself uses the group.
    symmetrize_forces = use_symmetry and symmetry_mode != "off"

    accelerated_self_consistent_soc = False
    if _accelerated_self_consistent_soc_eligible(translation):
        try:
            selection_probe = resolve_backend(arguments.backend, translation.problem)
        except (BackendUnavailableError, NotImplementedError):
            selection_probe = None
        accelerated_self_consistent_soc = (
            selection_probe is not None and selection_probe.selected == "cupy"
        )

    unsupported = _reference_only_features(
        translation, accelerated_backend_selected=accelerated_self_consistent_soc
    )
    if unsupported:
        features = ", ".join(unsupported)
        if arguments.backend in ("cupy", "native"):
            print(
                f"Input error: {features} is not implemented on the accelerated "
                f"path, so --backend {arguments.backend} cannot run this input; "
                "use --backend scipy (or auto) to run it with the reference "
                "solver",
                file=sys.stderr,
            )
            return 2
        problem = translation.problem
        reference_uses_gpu = (
            problem.periodic_cell is not None
            or problem.has_kpoint_sampling
            or problem.bands is not None
            # the non-collinear driver hands its eigenproblem to the k-point device workers, also
            # for an isolated system
            or problem.scf.noncollinear
        )
        if resources.device == "gpu" and not reference_uses_gpu:
            # Only the k-point drivers, band paths and the non-collinear driver of the reference
            # solver have a GPU path; anything else would run on the CPU under a request for the GPU.
            print(
                f"Input error: {features} cannot run on the GPU here (the accelerated "
                "path does not cover it, or CuPy is not usable), and the reference "
                "solver it would fall back to uses no GPU; use --device cpu or auto",
                file=sys.stderr,
            )
            return 2
        print(
            f"NOTE: {features} is not implemented on the accelerated path; "
            "running the reference parsec_python.cli solver instead",
            file=sys.stderr,
        )
        return reference_main(_reference_argv(arguments))

    if accelerated_self_consistent_soc and (
        translation.relaxation is not None
        or translation.molecular_dynamics is not None
        or translation.compute_forces
        or getattr(translation, "energy_refinement", None) is not None
    ):
        print(
            "Input error: Minimization, Molecular_Dynamics, Skip_Force: false and "
            "Energy_Refinement are not implemented together with self-consistent "
            "spin-orbit coupling",
            file=sys.stderr,
        )
        return 2
    if accelerated_self_consistent_soc:
        try:
            symmetry_mode = _resolve_symmetry_mode(
                arguments.symmetry,
                translation.ignore_symmetry,
                accelerated_self_consistent_soc=True,
            )
        except ValueError as error:
            print(f"Input error: {error}", file=sys.stderr)
            return 2
    symmetrize_forces = symmetrize_forces and symmetry_mode != "off"

    summary = summarize_translation(translation)
    if arguments.dry_run:
        try:
            pseudopotentials = load_pseudopotentials(
                translation.problem.pseudopotentials,
                xc_functional=translation.problem.scf.xc_functional,
            )
            domain_lines = domain_dry_run_lines(translation, pseudopotentials)
            selection = resolve_backend(arguments.backend, translation.problem)
            visible_gpus = _visible_gpus(resources) if selection.selected == "cupy" else None
        except (OSError, ValueError, BackendUnavailableError, NotImplementedError) as error:
            print(f"Validation error: {error}", file=sys.stderr)
            return 2
        print(summary)
        for line in domain_lines:
            print(line)
        for warning in translation.warnings:
            print(f"WARNING: {warning}")
        print(
            f"Backend: requested={selection.requested}, "
            f"selected={selection.selected}"
        )
        for line in resources.dry_run_lines(selection.selected, visible_gpus):
            print(line)
        print(
            "Finite-difference builder: "
            f"{selection.finite_difference_builder}"
        )
        print(f"Hartree backend: {selection.hartree_backend}")
        print(
            f"Symmetry: mode={symmetry_mode}; exact detection is deferred "
            "until grid construction"
        )
        for reason in selection.fallback_reasons:
            print(f"BACKEND FALLBACK: {reason}")
        print("Dry run successful; no grid or SCF calculation was started.")
        return 0

    if resources.device == "gpu":
        # A request for GPUs is held against what is there before any file
        # of the calculation is written: a number or a list against the GPUs
        # the process sees, and a plain request against the installation.
        # Whether a device answers is left to the preparation, which asks
        # the CUDA runtime beside its other work.
        try:
            if resources.names_gpus:
                _visible_gpus(resources)
            elif not _cupy_installed():
                raise BackendUnavailableError(
                    f"a GPU run was requested ({resources.device_source}) but CuPy is not "
                    "installed in this Python environment: install the CuPy build of your "
                    "CUDA version (python -m pip install cupy-cuda12x for CUDA 12), or run "
                    "with --device cpu"
                )
        except (ValueError, BackendUnavailableError) as error:
            print(f"Input error: {error}", file=sys.stderr)
            return 2

    input_directory = translation.source.parent
    symmetry_cache_directory = _symmetry_cache_directory(arguments)
    log_path = _resolve_output(
        arguments.log, input_directory, "parsec.out"
    )
    archive_path = _npz_path(
        _resolve_output(
            arguments.output,
            input_directory,
            "parsec_python_results.npz",
        )
    )
    protected_paths = {
        translation.source,
        *(item.path.resolve() for item in translation.problem.pseudopotentials.values()),
    }
    if log_path in protected_paths:
        print(f"Input error: refusing to overwrite input with log {log_path}", file=sys.stderr)
        return 2
    if not arguments.no_archive:
        if archive_path in protected_paths:
            print(
                f"Input error: refusing to overwrite input with archive {archive_path}",
                file=sys.stderr,
            )
            return 2
        if archive_path == log_path:
            print("Input error: --log and --output resolve to the same path", file=sys.stderr)
            return 2

    # The default rule of the sphere ran in the parser, before this clock:
    # its seconds belong to the calculation.  None for an input with a radius.
    started = time.perf_counter() - translation.domain_rule_seconds
    run_log = _RunLog(log_path, quiet=arguments.quiet)
    reporter = None
    try:
        with run_log as log:
            reporter = AcceleratedTextReporter(
                log.write,
                translation,
                symmetry_mode=symmetry_mode,
            )
            reporter.header()
            for warning in translation.warnings:
                log.write(f"WARNING: {warning}")
            if symmetry_note is not None:
                log.write(f"WARNING: {symmetry_note}")
            for note in restart_notes:
                log.write(note)
            log.write()
            if translation.relaxation is not None:
                log.write("\n".join(relaxation_header_lines(translation)))

            system = prepare_single_point(
                translation.problem,
                backend=arguments.backend,
                symmetry=symmetry_mode,
                symmetry_cache_directory=symmetry_cache_directory,
            )
            # Wavefunctions are much larger than all scalar observables and
            # are absent from the default archive.  Keep symmetry-sector
            # orbitals resident/lazy unless the user explicitly requests
            # their public full-grid array.  This does not affect SCF density,
            # energies, eigenvalues, or any normal text output.
            wavefunctions_wanted = bool(
                not arguments.no_archive
                and (
                    arguments.save_wavefunctions
                    or translation.output_all_states
                )
            )
            # A relaxation exports the orbitals of its last structure only.
            moving = (
                translation.relaxation is not None
                or translation.molecular_dynamics is not None
            )
            system.materialize_final_wavefunctions = wavefunctions_wanted and not moving
            reporter.setup(system)
            log.write("\n".join(_resource_report(resources, system)))
            if arguments.profile_operator:
                component_times = profile_hamiltonian_components(
                    system,
                    repeats=arguments.profile_repeats,
                )
                log.write(" Hamiltonian component microprofile [sec/apply]:")
                for name, value in component_times.items():
                    log.write(f" {name:30s} = {value:12.6f}")
                log.write()

            scf_started = time.perf_counter()
            if accelerated_self_consistent_soc:
                soc_projectors = tuple(
                    build_spin_orbit_projectors(
                        system.grid,
                        system.atoms,
                        system.pseudopotentials,
                        translation.problem.pseudopotentials,
                    )
                )
                result = run_self_consistent_soc(
                    system,
                    soc_projectors,
                    callback=reporter.reference.iteration_self_consistent_soc,
                )
                scf_completed = time.perf_counter()
                reporter.reference.finish_self_consistent_soc(
                    result, scf_completed - scf_started
                )
                reporting_completed = time.perf_counter()
                log.write(
                    f" Pre-SCF setup/reporting wall time [sec] : "
                    f"{scf_started - started:11.6f}"
                )
                log.write(
                    f" Post-SCF finalization/reporting [sec] : "
                    f"{reporting_completed - scf_completed:11.6f}"
                )
                log.write(
                    f" Total accelerated Python wall time [sec] : "
                    f"{time.perf_counter() - started:11.2f}"
                )
                if not arguments.no_archive:
                    saved = save_self_consistent_soc_result_archive(
                        archive_path,
                        result,
                        include_wavefunctions=(
                            arguments.save_wavefunctions
                            or translation.output_all_states
                        ),
                    )
                    log.write(f"Result archive: {saved}")
                log.write(f"Text log: {log_path}")
                return 0 if result.converged else 3


            forces = None
            outcome = None
            dynamics = None
            check_seconds = 0.0
            if translation.relaxation is not None:
                outcome = run_recorded_relaxation(
                    translation,
                    system,
                    ionic_backend(translation.potential_extrapolation),
                    reporter,
                    log.write,
                    directory=input_directory,
                    restart=relax_restart,
                    symmetrize=symmetrize_forces,
                )
                result = outcome.result
                forces = outcome.records[-1].forces
                if wavefunctions_wanted:
                    materialize_wavefunctions(outcome.system, result)
                scf_completed = time.perf_counter()
            elif translation.molecular_dynamics is not None:
                dynamics = run_recorded_molecular_dynamics(
                    translation,
                    system,
                    ionic_backend(translation.potential_extrapolation),
                    reporter,
                    log.write,
                    directory=input_directory,
                    restart=dynamics_restart,
                )
                result, forces = dynamics.result, dynamics.forces
                if wavefunctions_wanted:
                    materialize_wavefunctions(dynamics.system, result)
                scf_completed = time.perf_counter()
            elif translation.compute_forces:
                result, forces = run_single_point_with_forces(
                    system, ionic_backend(), reporter, symmetrize=symmetrize_forces
                )
                scf_completed = time.perf_counter()
            else:
                result = run_scf(system, callback=reporter.iteration)
                scf_completed = time.perf_counter()
                # run_scf makes the optional check of the Hartree boundary
                # before it returns: it is finalization, not SCF.
                check_seconds = hartree_boundary_check_seconds(result)
                reporter.finish(result, scf_completed - scf_started - check_seconds)
            reporting_completed = time.perf_counter()
            refinement = None
            if getattr(translation, "energy_refinement", None) is not None:
                # One non-self-consistent evaluation on the converged
                # orbitals, after everything the SCF reports; of the last
                # structure where the atoms moved.
                from parsec_python.Energy.refinement import refinement_report_lines
                from parsec_python.Output import RYDBERG_TO_EV

                from .refinement import refine_energy

                announce, progress = refinement_progress(
                    log.write, translation.energy_refinement
                )
                announce()
                refinement = refine_energy(
                    system if outcome is None else outcome.system,
                    result,
                    translation.energy_refinement,
                    progress=progress,
                )
                log.write(
                    "\n".join(refinement_report_lines(refinement, RYDBERG_TO_EV))
                )
            log.write(
                f" Pre-SCF setup/reporting wall time [sec] : "
                f"{scf_started - started:11.6f}"
            )
            log.write(
                f" Post-SCF finalization/reporting [sec] : "
                f"{reporting_completed - scf_completed + check_seconds:11.6f}"
            )
            if refinement is not None:
                log.write(
                    f" Post-SCF energy refinement [sec] : "
                    f"{time.perf_counter() - reporting_completed:11.6f}"
                )
            log.write(
                f" Total accelerated Python wall time [sec] : "
                f"{time.perf_counter() - started:11.2f}"
            )
            if not arguments.no_archive:
                saved = save_result_archive(
                    archive_path,
                    result,
                    include_wavefunctions=(
                        arguments.save_wavefunctions
                        or translation.output_all_states
                    ),
                    # A plain single point keeps the call it always made.
                    **{
                        name: value
                        for name, value in (
                            ("forces", forces),
                            ("relaxation", outcome),
                            ("dynamics", dynamics),
                            ("refinement", refinement),
                        )
                        if value is not None
                    },
                )
                log.write(f"Result archive: {saved}")
            log.write(f"Text log: {log_path}")
            if dynamics is not None:
                return 3 if dynamics.scf_failed else 0
            return relaxation_exit_code(
                result,
                outcome,
                manual=(
                    translation.relaxation is not None
                    and translation.relaxation.method == "manual"
                ),
            )
    except KeyboardInterrupt:
        if not getattr(run_log, "failure_reported", False) and not arguments.quiet:
            print("Calculation interrupted by user.", file=sys.stderr)
        return 130
    except Exception as error:
        if arguments.debug:
            raise
        if not getattr(run_log, "failure_reported", False):
            print(
                f"Calculation failed: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        return 1
    finally:
        # The reporter may run a host thread beside the SCF: it ends here
        # where the calculation did not reach its report.
        if reporter is not None:
            reporter.close()


__all__ = ["main"]
