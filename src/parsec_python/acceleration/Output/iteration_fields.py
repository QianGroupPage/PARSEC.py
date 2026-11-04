"""The fields of one accelerated SCF iteration on the host, for ``parsec.dat``.

With ``Save_Intermediate_Charge_Density`` PARSEC writes ``parsec.dat`` after
every SCF iteration (``parsec.F90`` lines 1381-1393).  The reference loop
hands the fields of each iteration to ``iteration_state_callback`` as an
:class:`parsec_python.SCF.single_point.SCFIterationFields`; on the
accelerated path its density and mixed potential are already full-grid host
arrays (the loop expands the fields it keeps per symmetry orbit at that
boundary), while its orbitals are those of the eigensolver: device arrays,
or the sector bases of one or several devices.  :func:`host_iteration_fields`
brings them to the host as one full-grid matrix, as the result of the last
iteration is brought there (:func:`..SCF.single_point.materialize_wavefunctions`),
where the file holds wavefunctions, and drops them where it holds none.
Nothing of the eigensolver state changes, but for sector bases that hold the
opt-in Ritz hand-over (``PARSEC_CUPY_RITZ_HANDOVER``), which are rotated into
their Ritz vectors first, as for the final orbitals.  A basis that several
devices share cannot be exported as one host array (as for the final
orbitals): the file of such an iteration holds no wavefunctions, and the
log says so once.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np


def host_wavefunctions(system, wavefunctions):
    """``wavefunctions`` of the accelerated ``system`` as a host matrix."""

    if wavefunctions is None or isinstance(wavefunctions, np.ndarray):
        return wavefunctions
    if system.backend_info.selected != "cupy":
        return np.asarray(wavefunctions, dtype=np.float64)
    from ..backends.cupy import require_cupy, synchronize

    cp, _ = require_cupy()
    to_host = getattr(wavefunctions, "to_full_host", None)
    to_device = getattr(wavefunctions, "to_full_device", None)
    if getattr(wavefunctions, "sector_handovers", None):
        ensure = getattr(
            getattr(system.backend, "symmetry_eigensolver", None),
            "ensure_ritz_vectors",
            None,
        )
        if callable(ensure):
            ensure()
    synchronize()
    try:
        if callable(to_host):
            host = to_host()
        elif callable(to_device):
            host = cp.asnumpy(to_device())
        else:
            host = cp.asnumpy(wavefunctions)
    except RuntimeError:
        # A sector basis that several devices share (or the sectors of other
        # MPI ranks): parsec.dat then notes that it keeps no orbitals.
        return None
    synchronize()
    return np.asarray(host, dtype=np.float64)


def host_iteration_fields(system, state, *, wavefunctions: bool):
    """``state`` with host orbitals (``wavefunctions``) or with none.

    ``state`` is the :class:`SCFIterationFields` of one iteration of the
    accelerated loop of ``system``; its channels are converted alike.
    """

    channels = getattr(state, "spin", None)
    if channels is not None:
        channels = tuple(
            replace(
                channel,
                wavefunctions=(
                    host_wavefunctions(system, channel.wavefunctions)
                    if wavefunctions
                    else None
                ),
            )
            for channel in channels
        )
    return replace(
        state,
        wavefunctions=(
            host_wavefunctions(system, state.wavefunctions) if wavefunctions else None
        ),
        **({} if channels is None else {"spin": channels}),
    )


def iteration_files_callback(files, system, problem):
    """The ``iteration_state_callback`` that writes the files of every iteration.

    ``files`` is the :class:`parsec_python.Output.output_files.RunOutputFiles`
    of the run; ``None`` where it asks for no fields of the iterations.
    """

    if not files.wants_iteration_fields:
        return None
    grid = system.grid

    def write(state) -> None:
        files.iteration_fields(
            host_iteration_fields(system, state, wavefunctions=files.selection_has_states),
            grid,
            problem,
        )

    return write


__all__ = ["host_iteration_fields", "host_wavefunctions", "iteration_files_callback"]
