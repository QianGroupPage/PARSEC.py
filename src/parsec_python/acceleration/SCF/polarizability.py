"""The seven field loops of ``Polarizability`` on the accelerated SCF loop.

:func:`parsec_python.SCF.polarizability.run_polarizability` runs the loops of
``polar.f90`` (fields ``0, +x, -x, +y, -y, +z, -z``) on a reference prepared
system: for each field it replaces the local ionic potential of that system
by ``V_ion + 2 F d.r`` and calls an ``scf_runner``.  The runner here takes
that reference system with the field into the accelerated system, which
keeps every execution object it holds (the Hamiltonian on the devices, the
eigensolver and its saved subspace, the Hartree solver): the field enters
the loop only through the ionic potential, which the SCF loop reads from the
reference system in every iteration (``V_in = V_ion + V_H + V_xc``).

Between loops PARSEC keeps the wavefunctions and sets ``pot%vnew =
pot%vold`` and ``solver%firstfilt`` (``parsec.F90`` lines 1650-1722): the
first eigenproblem of the next loop is posed with the *input* potential of
the last iteration, the old Hartree potential starts its first Poisson
solve, and the saved subspace, which stays on its devices, starts a
SUBSPACE filter (:func:`next_field_warm_start`).  The forces of the last
loop (field along ``-z``) are the accelerated forces of that loop, without
symmetrization and without a force of the field on the nuclei, as in
PARSEC.

``Polarizability`` uses no point group (``parsec.F90``; the parser sets no
symmetry), so the loops run on the full grid, or on the one sector of the
identity that several devices share (``PARSEC_CUPY_SINGLE_SECTOR``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from parsec_python.SCF import SCFWarmStart, state_for_new_potential


def field_runner(system, *, iteration_state_callback: Callable | None = None):
    """An ``scf_runner`` of ``run_polarizability`` for the accelerated ``system``.

    It is called with the reference system of one field and returns the
    accelerated result of that loop.  ``iteration_state_callback`` goes to
    every loop (``Save_Intermediate_Charge_Density``).
    """

    from ..driver import run_scf

    keywords = (
        {}
        if iteration_state_callback is None
        else {"iteration_state_callback": iteration_state_callback}
    )

    def run(reference, *, callback=None, warm_start=None):
        return run_scf(
            replace(system, reference=reference),
            callback=callback,
            warm_start=warm_start,
            **keywords,
        )

    return run


def next_field_warm_start(system):
    """The ``warm_start_builder`` of ``run_polarizability`` for ``system``.

    ``pot%vnew = pot%vold``: the input potential of the last iteration, the
    last Hartree potential and the saved subspaces of the eigensolver of
    ``system``, marked for a new SCF loop on their devices (one per spin
    channel of a spin-polarized loop).
    """

    def build(result) -> SCFWarmStart:
        solver = system.eigenproblem_solver
        channels = getattr(result, "spin", None)
        if channels is not None:
            for_channel = getattr(solver, "for_spin_channel", None)
            return SCFWarmStart(
                input_potential=tuple(
                    np.asarray(channel.input_effective_potential) for channel in channels
                ),
                hartree_potential=np.asarray(result.hartree_potential),
                eigensolver_state=tuple(
                    for_channel(index).start_new_scf_loop()
                    if callable(for_channel)
                    else state_for_new_potential(channel.eigensolver_state)
                    for index, channel in enumerate(channels)
                ),
            )
        restart = getattr(solver, "start_new_scf_loop", None)
        return SCFWarmStart(
            input_potential=np.asarray(result.input_effective_potential),
            hartree_potential=np.asarray(result.hartree_potential),
            eigensolver_state=(
                restart()
                if callable(restart)
                else state_for_new_potential(result.eigensolver_state)
            ),
        )

    return build


def field_forces(system):
    """The forces of the loop of one field: those of the accelerated driver.

    Called with the reference system of that loop and its result, before
    the eigensolver of ``system`` solves again.
    """

    from ..driver import compute_forces

    def forces(reference, result, *, symmetrize=False, operations=None):
        return compute_forces(
            replace(system, reference=reference),
            result,
            symmetrize=symmetrize,
            operations=operations,
        )

    return forces


__all__ = ["field_forces", "field_runner", "next_field_warm_start"]
