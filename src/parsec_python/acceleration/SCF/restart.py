"""``Restart_From_Wfndat`` on the accelerated SCF loop.

:func:`parsec_python.SCF.restart.load_wfndat_restart` reads ``wfn.dat``
(``restart.F90``) and returns the warm start of the first SCF loop: the
stored potential, the Hartree start and an eigensolver state whose subspace
begins with the stored orbitals, completed to the working size of the run
by random columns orthogonal to them (``eig_init``: the first eigenproblem
is a SUBSPACE filter of the stored states).  That state is a host state of
the reference eigensolver.

:func:`accelerated_restart_start` hands the same subspace to the eigensolver
of an accelerated system.  On the CPU backends the reference eigensolver
solves and takes the state as it is.  On the GPU the subspace is uploaded
into the persistent CuPy eigensolver of the full grid (one per spin
channel), as the state its next solve continues from, with the stored
levels as its Ritz values and ``firstfilt`` set: the first pass is the
SUBSPACE filter of the reference loop, on the device.

A file of this port holds full-grid orbitals of one representation
(``nrep = 1``, errors 321-329), so a run restarted from it solves on the
full grid: the accelerated CLI switches the symmetry sectors off for it.
An eigensolver of sectors (also the one sector of the identity of
``PARSEC_CUPY_SINGLE_SECTOR``) takes no stored subspace and is refused.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np


def _upload_subspace(scf_solver, state, settings):
    """Seed ``scf_solver`` (a CuPy SCF eigensolver) with the host ``state``."""

    from ..backends.cupy import require_cupy
    from ..Eigensolvers.eigval import CuPyEigvalDeviceState, CuPyEigvalSolver
    from ..Eigensolvers.subspace import DeviceSubspaceState

    cp, _ = require_cupy()
    if scf_solver.solver is None:
        # The solver the first call would make (CuPySCFEigensolver.__call__).
        scf_solver.solver = CuPyEigvalSolver(
            scf_solver.operator,
            settings=settings,
            retain_vectors_on_device=True,
            compute_subspace_residuals=False,
        )
    saved = state.subspace
    vectors = np.asarray(saved.vectors, dtype=np.float64)
    dimension, working = vectors.shape
    if dimension != int(scf_solver.operator.shape[0]):
        raise ValueError("the stored subspace does not match the grid of this run")
    device_state = CuPyEigvalDeviceState(
        operator_dimension=dimension,
        requested_states=int(state.requested_states),
        working_states=int(working),
        initial_method=str(state.initial_method),
        subspace=DeviceSubspaceState(
            operator_dimension=dimension,
            working_states=int(working),
            eigenvalues=cp.asarray(saved.eigenvalues, dtype=cp.float64),
            vectors=cp.asarray(vectors, dtype=cp.float64, order="F"),
            filter_lower_bound=None,
            first_filter=True,
            filters_completed=0,
        ),
        solves_completed=1,
    )
    scf_solver.solver._state = device_state
    return device_state


def accelerated_restart_start(system, warm_start):
    """The warm start of ``Restart_Run`` for the accelerated ``system``.

    A warm start without an eigensolver state (``parsec.dat``) is returned
    as it is.  With the stored subspace of ``wfn.dat`` the state is moved
    into the eigensolver of ``system`` (see the module text).
    """

    state = getattr(warm_start, "eigensolver_state", None)
    solver = getattr(system, "eigenproblem_solver", None)
    if warm_start is None or state is None or solver is None:
        return warm_start
    if getattr(system.backend, "symmetry_eigensolver", None) is not None:
        raise NotImplementedError(
            "Restart_From_Wfndat seeds the eigensolver of the full grid; this run "
            "solves in symmetry sectors (also the one sector of "
            "PARSEC_CUPY_SINGLE_SECTOR=on), which take no stored subspace"
        )
    from parsec_python.SCF.single_point import (
        eigval_settings_for,
        first_diagonalization_tolerance,
    )

    eigensolver = system.input.eigensolver
    settings = eigval_settings_for(
        eigensolver,
        eigensolver.filter_degree,
        tolerance=first_diagonalization_tolerance(eigensolver),
    )
    if isinstance(state, tuple):
        channel_solver = getattr(solver, "for_spin_channel", None)
        if not callable(channel_solver):
            raise NotImplementedError("this eigensolver keeps no subspace per spin channel")
        seeded = tuple(
            _upload_subspace(channel_solver(index), item, settings)
            for index, item in enumerate(state)
        )
    else:
        seeded = _upload_subspace(solver, state, settings)
    return replace(warm_start, eigensolver_state=seeded)


__all__ = ["accelerated_restart_start"]
