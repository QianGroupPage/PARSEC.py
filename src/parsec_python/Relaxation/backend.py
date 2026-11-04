"""The operations an ionic loop needs from an execution path.

A relaxation and a molecular dynamics run repeat the same four operations
for every structure: solve the electronic structure, evaluate the forces,
rebuild what depends on the atom positions, and hand the next SCF loop what
it continues from.  How they are executed is the business of the readable
NumPy path or of an accelerated path; an :class:`IonicBackend` names them so
that the loops in :mod:`parsec_python.Relaxation` and
:mod:`parsec_python.MolecularDynamics` are written once.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

from ..models import Atom


@dataclass(frozen=True)
class IonicBackend:
    """The four operations of one execution path.

    ``run_scf(system, callback=..., warm_start=...)`` returns the SCF result
    of a prepared system; ``compute_forces(system, result, symmetrize=...,
    operations=...)`` returns a :class:`parsec_python.Forces.ForceResult`;
    ``update_geometry(system, atoms)`` returns the prepared system of the
    moved atoms on the same grid; ``warm_start(previous_system,
    previous_result, system)`` returns what the next SCF loop takes over, in
    the form ``run_scf`` accepts.  ``restart_scf(system, result)``, where
    given, is what a second SCF loop of the same structure continues from,
    without any extrapolation (``Relaxation_SCF_Upscale``); without it the
    loop takes ``warm_start(system, result, system)``.  ``run_scf`` takes
    ``convergence_criterion=`` where the relaxation loosens the criterion.
    ``symmetry_operations(system)``, where given, returns the operations a
    relaxation keeps its structure on (a periodic cell, see
    :func:`fortran_periodic_backend`); without it they are the point-group
    operations of the atoms of an isolated system.
    """

    run_scf: Callable[..., object]
    compute_forces: Callable[..., object]
    update_geometry: Callable[[object, Sequence[Atom]], object]
    warm_start: Callable[[object, object, object], object]
    restart_scf: Callable[[object, object], object] | None = None
    symmetry_operations: Callable[[object], object] | None = None
    # Andrew Grieder's: whether a moved structure continues from the one before (the report says so).
    warm_starts: bool = True
    # Whether the atoms of a moved structure keep their other fields (the initial moment of a non-collinear
    # calculation; Andrew Grieder's loops) or are new atoms of symbol and position (the Fortran-derived
    # routines of the parity round, ``fortran_spinor_backend``).
    keep_atom_fields: bool = True


def moved_atoms(atoms, positions, keep_fields: bool = True) -> tuple:
    """The atoms at new positions.

    ``keep_fields``: each keeps its other fields (the initial moment of a non-collinear calculation; Andrew
    Grieder's routines); else new atoms of symbol and position (the Fortran-derived routines).
    """

    if keep_fields:
        return tuple(replace(atom, position=position) for atom, position in zip(atoms, positions))
    return tuple(Atom(atom.symbol, position) for atom, position in zip(atoms, positions))


def reference_backend(potential_extrapolation: str = "none") -> IonicBackend:
    """The readable full-grid NumPy/SciPy path.

    ``potential_extrapolation`` is how the SCF loop of moved atoms starts
    (``Potential_Extrapolation``; see
    :func:`parsec_python.SCF.warm_start_for_moved_atoms`).
    """

    from functools import partial

    from ..Forces import compute_forces
    from ..SCF import run_scf, update_geometry, warm_start_for_moved_atoms
    from ..SCF.single_point import extrapolation_keywords

    keywords = extrapolation_keywords(potential_extrapolation)
    return IonicBackend(
        run_scf=run_scf,
        compute_forces=compute_forces,
        update_geometry=update_geometry,
        warm_start=partial(warm_start_for_moved_atoms, **keywords),
        restart_scf=lambda system, result: warm_start_for_moved_atoms(system, result, system),
    )


def wire_slab_backend(
    k_sampling=None, potential_extrapolation: str = "none"
) -> IonicBackend:
    """The readable path of a wire or slab (``Boundary_Conditions`` wire/slab).

    :func:`fortran_periodic_backend` with the operations of the wire or slab: moved
    atoms from :func:`parsec_python.SCF.wire_slab.update_wire_slab_geometry`
    (``back_to_cell``, ``isolat``, ``ionpot_wire``/``ionpot_slab``,
    ``forceion_wire``/``ewald_slab``), forces from
    :func:`parsec_python.Forces.compute_forces` (``forcloc_wire``/
    ``forcloc_slab``), the structure kept on the space group the input reader
    found for the wire (a slab has none: Fortran switches its symmetry off).
    """

    from ..Forces.wire_slab import wire_slab_symmetry_operations
    from ..SCF.wire_slab import update_wire_slab_geometry

    def operations(system):
        found = wire_slab_symmetry_operations(system)
        if found is None:
            from ..Forces.symmetry import AtomSymmetryOperation
            import numpy as np

            return (
                AtomSymmetryOperation(
                    matrix=np.eye(3), image=np.arange(len(system.atoms), dtype=np.int64)
                ),
            )
        return found

    return fortran_periodic_backend(
        k_sampling,
        potential_extrapolation,
        update_geometry=update_wire_slab_geometry,
        symmetry_operations=operations,
    )


def fortran_periodic_backend(
    k_sampling=None,
    potential_extrapolation: str = "none",
    *,
    update_geometry=None,
    symmetry_operations=None,
) -> IonicBackend:
    """The readable path of a periodic (bulk) cell with a fixed cell (Fortran-derived routines).

    ``periodic_backend`` of the Fortran parity round, for ``Use_Fortran_Routines``; the default is Andrew Grieder's
    :func:`periodic_backend`.

    ``k_sampling`` is ``(k_points, weights)`` of a k-point calculation, or
    ``None`` at Gamma.  The SCF loop at Gamma is :func:`parsec_python.SCF.run_scf`
    (with collinear spin :func:`parsec_python.SCF.run_spin_scf`, started from
    the periodic spin densities of
    :func:`parsec_python.SCF.pbc.with_initial_spin_densities`); with k-points
    it is :func:`parsec_python.SCF.run_scf_kpoints`, keeping its orbitals
    (with collinear spin :func:`parsec_python.SCF.kpoints.run_scf_kpoints_spin_polarized`,
    started from the two channel potentials of the structure before).
    Forces come from :func:`parsec_python.Forces.compute_forces` (which sends
    a periodic system to :mod:`parsec_python.Forces.periodic`), moved atoms
    from :func:`parsec_python.SCF.pbc.update_periodic_geometry`.  As in
    ``parsec.F90`` the next SCF loop starts from the last mixed potential
    with the ionic part replaced (and at Gamma from the saved orbitals); a
    k-point loop rebuilds its subspace.  ``update_geometry`` and
    ``symmetry_operations`` replace those of a bulk cell
    (:func:`wire_slab_backend`).
    """

    from functools import partial

    from ..Forces import compute_forces, periodic_force_operations
    from ..SCF import run_scf, warm_start_for_moved_atoms
    from ..SCF.pbc import update_periodic_geometry, with_initial_spin_densities
    from ..SCF.single_point import extrapolation_keywords

    keywords = extrapolation_keywords(potential_extrapolation)
    warm_start = partial(warm_start_for_moved_atoms, **keywords)
    if update_geometry is None:
        update_geometry = update_periodic_geometry

    def bulk_operations(system):
        # The space group where the problem carries one (symm_forces.f90,
        # non-symmorphic operations included), else the point operations.
        k_points = None if k_sampling is None else k_sampling[0]
        return periodic_force_operations(system, k_points)

    operations = bulk_operations if symmetry_operations is None else symmetry_operations

    if k_sampling is None:

        def run_gamma(system, *, warm_start=None, **options):
            if warm_start is None:
                system = with_initial_spin_densities(system)
            return run_scf(system, warm_start=warm_start, **options)

        return IonicBackend(
            run_scf=run_gamma,
            compute_forces=compute_forces,
            update_geometry=update_geometry,
            warm_start=warm_start,
            restart_scf=lambda system, result: warm_start_for_moved_atoms(
                system, result, system
            ),
            symmetry_operations=operations,
        )

    from dataclasses import replace

    import numpy as np

    from ..SCF import run_scf_kpoints

    k_points, k_weights = k_sampling

    def with_criterion(system, convergence_criterion):
        if convergence_criterion is None:
            return system
        scf = replace(system.input.scf, convergence_criterion=float(convergence_criterion))
        return replace(system, input=replace(system.input, scf=scf))

    def spin_warm_start(previous_system, previous_result, system):
        # parsec.F90 after domove, per spin channel: vnew - vion_old + vion_new.
        # (Potential_Extrapolation starts a spin-polarized loop as "none", as
        # for an isolated system.)
        return tuple(
            np.asarray(potential, dtype=np.float64)
            - np.asarray(previous_system.ionic_potential)
            + np.asarray(system.ionic_potential)
            for potential in previous_result.next_potentials
        )

    def kpoint_warm_start(previous_system, previous_result, system):
        if getattr(previous_result, "next_potentials", None) is not None:
            return spin_warm_start(previous_system, previous_result, system)
        return warm_start(previous_system, previous_result, system)

    def run_kpoints(
        system,
        *,
        callback=None,
        warm_start=None,
        convergence_criterion=None,
        iteration_state_callback=None,
    ):
        # Save_Intermediate_*: the k-point loops call it after every iteration.
        state = (
            {}
            if iteration_state_callback is None
            else {"iteration_state_callback": iteration_state_callback}
        )
        if system.input.scf.spin_polarized:
            from ..SCF.kpoints import run_scf_kpoints_spin_polarized

            # Collinear spin with k-points: the warm start is the pair of
            # channel potentials of spin_warm_start.
            return run_scf_kpoints_spin_polarized(
                with_criterion(system, convergence_criterion),
                k_points,
                k_weights,
                callback=callback,
                keep_wavefunctions=True,
                initial_potentials=warm_start,
                **state,
            )
        if convergence_criterion is not None:
            scf = replace(system.input.scf, convergence_criterion=float(convergence_criterion))
            system = replace(system, input=replace(system.input, scf=scf))
        initial = None
        if warm_start is not None:
            initial = np.asarray(warm_start.input_potential, dtype=np.float64)
            guess = getattr(warm_start, "density_guess", None)
            if guess is not None:
                # Potential_Extrapolation: the Hartree and XC potentials of
                # the extrapolated density replace those of the last output.
                hartree = system.solve_hartree(
                    guess, initial_potential=warm_start.hartree_potential
                ).potential
                initial = (
                    initial
                    + hartree
                    + system.evaluate_xc(guess).potential
                    - np.asarray(warm_start.hxc_reference, dtype=np.float64)
                )
        return run_scf_kpoints(
            system,
            k_points,
            k_weights,
            callback=callback,
            initial_potential=initial,
            keep_wavefunctions=True,
            **state,
        )

    return IonicBackend(
        run_scf=run_kpoints,
        compute_forces=compute_forces,
        update_geometry=update_geometry,
        warm_start=kpoint_warm_start,
        restart_scf=lambda system, result: (
            spin_warm_start(system, result, system)
            if getattr(result, "next_potentials", None) is not None
            else warm_start_for_moved_atoms(system, result, system)
        ),
        symmetry_operations=operations,
    )


def fortran_spinor_backend(k_sampling=None, *, periodic: bool = False) -> IonicBackend:
    """The readable path of the spin-orbit and non-collinear drivers (Fortran-derived routines).

    ``spinor_backend`` of the Fortran parity round, for ``Use_Fortran_Routines``; the default is Andrew Grieder's
    :func:`spinor_backend`.

    ``SO_from_scratch``/``SCF_SO`` (also with collinear spin) and
    ``Non_Collinear_magnetism`` (with or without spin-orbit species), for an
    isolated system or a periodic cell (``periodic``; ``k_sampling`` is
    ``(k_points, weights)``, ``None`` for Gamma alone).  The SCF loop is the
    driver of :mod:`parsec_python.SCF` the single point runs (the k-point
    forms keep the spinors of every k-point), the forces those of
    :func:`parsec_python.Forces.spin_orbit.compute_spin_orbit_forces`, moved
    atoms :func:`parsec_python.SCF.update_geometry` or
    :func:`parsec_python.SCF.pbc.update_periodic_geometry`.  As in
    ``parsec.F90`` every structure after the first starts from the last mixed
    potential of the structure before with its ionic part replaced
    (``vnew - vion_old + vion_new`` for every spin component; a
    non-collinear loop keeps its exchange-correlation field, the direction
    of the magnetization), with Andrew Grieder's warm start
    (:func:`parsec_python.SCF.warm_start_for_moved_atoms`: also the Hartree
    potential and the eigensubspaces the loop exports, those of every
    k-point included), which replaced the potential-only start of the parity round
    (25-45 % fewer SCF iterations, the same trajectories).  The
    first structure of a ``Restart_Run`` from a spin-orbit ``parsec.dat``
    starts from the potentials of the file (the warm start that
    :func:`parsec_python.SCF.restart.with_first_warm_start` hands over, as
    the single point of the spin-orbit drivers takes it).
    """

    import numpy as np

    from ..Eigensolvers.perturbative_soc import build_spin_orbit_projectors
    from ..Forces import compute_forces, periodic_force_operations
    from ..SCF import update_geometry, warm_start_for_moved_atoms
    from ..SCF.kpoints_soc import (
        run_self_consistent_soc_kpoints,
        run_self_consistent_soc_kpoints_spin_polarized,
    )
    from ..SCF.noncollinear import (
        run_self_consistent_noncollinear,
        run_self_consistent_noncollinear_kpoints,
    )
    from ..SCF.pbc import update_periodic_geometry
    from ..SCF.self_consistent_soc import run_self_consistent_soc
    from ..SCF.self_consistent_soc_spin_polarized import run_self_consistent_soc_spin_polarized

    def run_spinor_scf(
        system,
        *,
        callback=None,
        warm_start=None,
        convergence_criterion=None,
        iteration_state_callback=None,
    ):
        from dataclasses import replace

        # Save_Intermediate_*: every spinor loop calls it after each iteration.
        if iteration_state_callback is not None:
            start_state = {"iteration_state_callback": iteration_state_callback}
        else:
            start_state = {}
        if convergence_criterion is not None:
            scf = replace(system.input.scf, convergence_criterion=float(convergence_criterion))
            system = replace(system, input=replace(system.input, scf=scf))
        settings = system.input.scf
        start = {}
        if isinstance(warm_start, _MovedStructureStart):
            # Andrew Grieder's warm start of a moved structure (or of the same one, Relaxation_SCF_Upscale)
            start["warm_start"] = warm_start.start
        elif isinstance(warm_start, SpinorWarmStart):
            # a start built from a finished loop (the field loops of Polarizability)
            start["initial_potential"] = warm_start.potential
            if warm_start.eigensolver_state is not None and not (
                periodic or settings.noncollinear
            ):
                # The isolated spin-orbit loops filter the saved spinors.
                start["initial_eigensolver_state"] = warm_start.eigensolver_state
        elif warm_start is not None and not settings.noncollinear:
            # Restart_Run from a spin-orbit parsec.dat (restart_run.F90 with
            # stype(2) = 2): the potentials of the two spin components.
            from ..SCF.restart import spin_orbit_start

            start["initial_potential"] = spin_orbit_start(warm_start, settings.spin_polarized)
        if periodic:
            if k_sampling is None:
                k_points, weights = np.zeros((1, 3)), np.ones(1)
            else:
                k_points, weights = k_sampling
            if settings.noncollinear:
                driver = run_self_consistent_noncollinear_kpoints
            elif settings.spin_polarized:
                driver = run_self_consistent_soc_kpoints_spin_polarized
            else:
                driver = run_self_consistent_soc_kpoints
            return driver(
                system,
                k_points,
                weights,
                callback=callback,
                keep_wavefunctions=True,
                **start,
                **start_state,
            )
        projectors = tuple(
            build_spin_orbit_projectors(
                system.grid, system.atoms, system.pseudopotentials, system.input.pseudopotentials
            )
        )
        if settings.noncollinear:
            return run_self_consistent_noncollinear(
                system, projectors, callback=callback, **start, **start_state
            )
        if settings.spin_polarized:
            return run_self_consistent_soc_spin_polarized(
                system, projectors, callback=callback, **start, **start_state
            )
        return run_self_consistent_soc(
            system, projectors, callback=callback, **start, **start_state
        )

    def operations(system):
        k_points = None if k_sampling is None else k_sampling[0]
        return periodic_force_operations(system, k_points)

    return IonicBackend(
        run_scf=run_spinor_scf,
        compute_forces=compute_forces,
        update_geometry=update_periodic_geometry if periodic else update_geometry,
        warm_start=lambda previous_system, result, system: _MovedStructureStart(
            warm_start_for_moved_atoms(previous_system, result, system)
        ),
        restart_scf=lambda system, result: _MovedStructureStart(warm_start_for_moved_atoms(system, result, system)),
        symmetry_operations=operations if periodic else None,
        keep_atom_fields=False,
    )


def periodic_backend(
    k_points=None, k_weights=None, potential_extrapolation: str = "none"
) -> IonicBackend:
    """The readable path for a periodic cell: the Gamma point, or a uniform k-point sampling.

    ``k_points`` and ``k_weights`` (the Monkhorst-Pack points of the input, equal weights) select the k-point driver
    :func:`parsec_python.SCF.kpoints.run_scf_kpoints`, which keeps no eigenvectors, so its SCF step also accumulates
    the nonlocal force from them; without them the Gamma-point loop runs and the forces are built from its orbitals.
    The forces are those of :func:`parsec_python.Forces.compute_periodic_forces`: no symmetrization (the structure is
    not reduced by space-group operations) and the net force removed, which is what Fortran does for a crystal.  The
    k-point loops start their eigenproblems cold after a move; the Gamma loop carries its eigensubspace over.
    """

    from functools import partial

    import numpy as np

    from ..Forces import compute_periodic_forces
    from ..SCF import run_scf, update_periodic_geometry, warm_start_for_moved_atoms
    from ..SCF.kpoints import run_scf_kpoints
    from ..SCF.single_point import extrapolation_keywords

    sampled = k_points is not None
    if sampled:
        k_points = np.asarray(k_points, dtype=float)
        k_weights = np.asarray(k_weights, dtype=float)

    def solve(system, *, callback=None, warm_start=None, convergence_criterion=None):
        if sampled:
            return run_scf_kpoints(
                system,
                k_points,
                k_weights,
                callback=callback,
                compute_forces=True,
                warm_start=warm_start,
                convergence_criterion=convergence_criterion,
            )
        return run_scf(
            system, callback=callback, warm_start=warm_start, convergence_criterion=convergence_criterion
        )

    def forces(system, result, *, symmetrize=False, operations=None):
        if symmetrize or operations is not None:
            raise ValueError("a periodic cell has no point-group symmetrization of its forces here")
        return compute_periodic_forces(system, result)

    keywords = extrapolation_keywords(potential_extrapolation)
    return IonicBackend(
        run_scf=solve,
        compute_forces=forces,
        update_geometry=update_periodic_geometry,
        warm_start=partial(warm_start_for_moved_atoms, **keywords),
        restart_scf=lambda system, result: warm_start_for_moved_atoms(system, result, system),
    )


def spinor_backend(k_points=None, k_weights=None, potential_extrapolation: str = "none") -> IonicBackend:
    """The readable path for spinor orbitals: self-consistent spin-orbit coupling and non-collinear magnetism.

    The drivers are those of ``SO_from_scratch`` (``run_self_consistent_soc``, with collinear spin
    ``run_self_consistent_soc_spin_polarized``) and ``Non_Collinear_magnetism`` (``run_self_consistent_noncollinear``),
    isolated, or for a periodic cell their k-point versions at the Gamma point (``k_points`` omitted) or on the
    unreduced uniform sampling ``k_points`` / ``k_weights``; the one the prepared system's settings ask for is run, and
    the spin-orbit projectors are built from its atoms at every structure.  The forces are those of
    :func:`parsec_python.Forces.compute_spinor_forces` (isolated, no symmetrization unless asked for) or
    :func:`parsec_python.Forces.compute_periodic_forces` (periodic, whose k-point result carries the nonlocal force
    accumulated by the SCF step).

    The SCF loop of a moved structure continues from the previous one: its mixed potential (for spin the two
    potentials, for a non-collinear run the exchange-correlation field with the moments) and the eigensubspaces of its
    orbitals.  ``potential_extrapolation`` is as for :func:`reference_backend` for plain spin-orbit coupling; the loops
    with spin take the carried potentials only.
    """

    import copy
    from functools import partial

    import numpy as np

    from ..Eigensolvers.perturbative_soc import build_spin_orbit_projectors
    from ..Forces import compute_periodic_forces, compute_spinor_forces
    from ..SCF import update_geometry, update_periodic_geometry, warm_start_for_moved_atoms
    from ..SCF.kpoints_soc import run_self_consistent_soc_kpoints, run_self_consistent_soc_kpoints_spin_polarized
    from ..SCF.noncollinear import run_self_consistent_noncollinear, run_self_consistent_noncollinear_kpoints
    from ..SCF.self_consistent_soc import run_self_consistent_soc
    from ..SCF.self_consistent_soc_spin_polarized import run_self_consistent_soc_spin_polarized
    from ..SCF.single_point import extrapolation_keywords

    if k_points is not None:
        k_points = np.asarray(k_points, dtype=float)
        k_weights = np.asarray(k_weights, dtype=float)

    def is_periodic(system) -> bool:
        return getattr(system.input, "periodic_cell", None) is not None

    def with_criterion(system, criterion):
        # the drivers read their criterion from the settings of the system: a copy with the loosened one
        loosened = copy.copy(system)
        scf = replace(system.input.scf, convergence_criterion=float(criterion))
        object.__setattr__(loosened, "input", replace(system.input, scf=scf))
        return loosened

    def solve(system, *, callback=None, warm_start=None, convergence_criterion=None):
        if convergence_criterion is not None:
            system = with_criterion(system, convergence_criterion)
        scf = system.input.scf
        if is_periodic(system):
            points, weights = (k_points, k_weights) if k_points is not None else (np.zeros((1, 3)), np.ones(1))
            if scf.noncollinear:
                driver = run_self_consistent_noncollinear_kpoints
            elif scf.spin_polarized:
                driver = run_self_consistent_soc_kpoints_spin_polarized
            else:
                driver = run_self_consistent_soc_kpoints
            return driver(system, points, weights, callback=callback, compute_forces=True, warm_start=warm_start)
        projectors = tuple(
            build_spin_orbit_projectors(
                system.grid, system.atoms, system.pseudopotentials, system.input.pseudopotentials
            )
        )
        if scf.noncollinear:
            return run_self_consistent_noncollinear(system, projectors, callback=callback, warm_start=warm_start)
        if scf.spin_polarized:
            return run_self_consistent_soc_spin_polarized(system, projectors, callback=callback, warm_start=warm_start)
        return run_self_consistent_soc(system, projectors, callback=callback, warm_start=warm_start)

    def forces(system, result, *, symmetrize=False, operations=None):
        if is_periodic(system):
            if symmetrize or operations is not None:
                raise ValueError("a periodic cell has no point-group symmetrization of its forces here")
            return compute_periodic_forces(system, result)
        return compute_spinor_forces(system, result, symmetrize=symmetrize, operations=operations)

    def update(system, atoms):
        return update_periodic_geometry(system, atoms) if is_periodic(system) else update_geometry(system, atoms)

    return IonicBackend(
        run_scf=solve,
        compute_forces=forces,
        update_geometry=update,
        warm_start=partial(warm_start_for_moved_atoms, **extrapolation_keywords(potential_extrapolation)),
        restart_scf=lambda system, result: warm_start_for_moved_atoms(system, result, system),
    )


@dataclass(frozen=True)
class SpinorWarmStart:
    """Where a spinor SCF loop starts (the field loops of ``Polarizability``, ``cli.py``).

    ``potential`` is the driver's ``initial_potential``, ``eigensolver_state``
    the subspace an isolated spin-orbit loop filters first (``None``: random
    vectors).
    """

    potential: object
    eigensolver_state: object | None = None


@dataclass(frozen=True)
class _MovedStructureStart:
    """The warm start of :func:`fortran_spinor_backend` for a moved structure (Andrew Grieder's
    :class:`parsec_python.SCF.single_point.SCFWarmStart`, told apart from the start of a ``Restart_Run``)."""

    start: object


def release_orbitals(result) -> None:
    """Drop the views a finished result holds on the eigensolver subspace.

    An accelerated eigensolver keeps the subspace the next SCF loop continues
    from; views of it in the result of the previous structure would only pin
    device memory.  A result with host orbitals is left as it is.
    """

    release = getattr(
        getattr(result, "wavefunctions", None), "release_intermediate_storage", None
    )
    if callable(release):
        release()


__all__ = [
    "IonicBackend",
    "SpinorWarmStart",
    "fortran_periodic_backend",
    "fortran_spinor_backend",
    "periodic_backend",
    "reference_backend",
    "release_orbitals",
    "spinor_backend",
    "wire_slab_backend",
]
