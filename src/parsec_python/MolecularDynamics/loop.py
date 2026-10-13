"""The atom movement loop of ``parsec.F90`` for molecular dynamics.

In the Fortran main program a molecular dynamics run is this loop (label 30
to label 80), with ``move%num`` starting at zero::

    30  self-consistent field, total energy and forces of clust%xatm
        if (step_num == 0) goto 80
        call moldyn(clust,mol_dynamic,pbc,move%num,bdev)   ! move%num + 1
        if (move%num > step_num) goto 80
        prepare the potentials for the new clust%xatm
        goto 30
    80  continue

``moldyn`` is therefore called ``Step_num + 1`` times, each time after a
self-consistent calculation, and the structure returned by the last call is
not calculated any more.  :func:`run_molecular_dynamics` is that loop with
the electronic structure behind a callable.  It is meant as the reference
for a driver that owns the self-consistent field, and it is what the tests
compare with the Fortran program; it does no input or output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .md_files import MoldynRecords, initial_records, step_records
from .moldyn import MolecularDynamicsState, MoldynStep, moldyn_step
from .random_numbers import UniformSource
from .settings import MolecularDynamicsSettings


@dataclass(frozen=True, eq=False)
class MolecularDynamicsRun:
    """Outcome of :func:`run_molecular_dynamics`.

    ``steps`` holds one :class:`MoldynStep` per call of ``moldyn`` and
    ``state`` the state after the last one.  ``evaluations`` counts the
    calls of the energy and force function.  ``records`` is the complete
    text of ``md_nrg.dat``, ``md_mech.dat``, ``atom.cor`` and ``movie.xyz``
    when species names were given, else ``None``.
    """

    steps: tuple[MoldynStep, ...]
    state: MolecularDynamicsState
    evaluations: int
    records: MoldynRecords | None


def run_molecular_dynamics(
    settings: MolecularDynamicsSettings,
    state: MolecularDynamicsState,
    evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]],
    *,
    random: UniformSource | None = None,
    species: Sequence[str] | None = None,
    first_positions: np.ndarray | None = None,
) -> MolecularDynamicsRun:
    """Run the molecular dynamics loop of ``parsec.F90``.

    ``evaluate(positions)`` returns the total energy in Ry and the forces in
    Ry/bohr (shape ``(atoms, 3)``) of a structure given in bohr.
    ``state`` comes from ``start_molecular_dynamics`` or
    ``restart_molecular_dynamics``.  ``random`` defaults to
    ``settings.random_source()``.  ``first_positions`` is the structure of
    the first evaluation when it differs from ``state.current_positions``,
    which happens only in a restart whose cluster PARSEC recentres.
    """
    if random is None:
        random = settings.random_source()
    positions = state.current_positions if first_positions is None else first_positions
    records = None
    if species is not None:
        records = initial_records(state, species, positions)
    steps: list[MoldynStep] = []
    evaluations = 0
    while True:
        total_energy, forces = evaluate(np.array(positions, dtype=float))
        evaluations += 1
        if settings.step_num == 0:
            break
        step = moldyn_step(settings, state, forces, total_energy, random=random)
        if records is not None:
            records = records + step_records(state, step, species)
        steps.append(step)
        state = step.state
        positions = step.positions
        if step.finished:
            break
    return MolecularDynamicsRun(
        steps=tuple(steps),
        state=state,
        evaluations=evaluations,
        records=records,
    )
