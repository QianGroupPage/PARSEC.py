"""PARSEC 1.5's restart of the first eigensolver in ionic loops (``Filter_Restart_After``).

``parsec.F90:2944-3026`` of PARSEC 1.5, after the atoms moved: when the number
of movements made is a multiple of ``Filter_Restart_After`` and
``Restart_Cheb_Remained_Times`` restarts are left, the next SCF loop starts
with the first eigensolver again (``eig_init = .false.``, the polynomial
degree of the input), and, while ``Remained_Times_Atomic_Chrg_Den`` density
resets are left, from the superposed atomic densities with new Hartree and
exchange-correlation potentials (``initchrg``, ``rho_hart``,
``vnew = vion + vxc + vhart``).  Both counters fall by one at each use
(defaults 5000, 3, 3).

Here a density reset is the cold start of the SCF loop (no warm start: the
superposed atomic density and a first solve, as for the first structure);
a restart without one keeps the potential the loop would have taken over and
drops only the saved subspace, so the first eigensolver starts from new
random vectors.  1.5 instead runs CHEBFF in its ``cont`` mode then
(``chebff.f90z:180-231``: the converged vectors kept, random ones for the
rest), which is not reproduced.  This module only decides; the ionic loops
(:mod:`parsec_python.Relaxation.ionic_loop`,
:mod:`parsec_python.MolecularDynamics.run`) apply it.

The warm starts the ionic backends hand over differ in type
(:mod:`parsec_python.Relaxation.backend`): an
:class:`parsec_python.SCF.single_point.SCFWarmStart` (with one subspace, a
pair for collinear spin, or a ``{k-point: state}`` dict), a wrapper around one
(``fortran_spinor_backend``), or the bare channel potentials of a spin
k-point loop, which rebuilds its subspace for every structure anyway.  A
backend whose warm start is not handled by :func:`drop_eigensolver_subspace`
supplies its own ``drop_subspace`` (``IonicBackend.drop_subspace``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace

import numpy as np


def drop_eigensolver_subspace(carried):
    """``carried`` with its saved eigensolver subspace dropped, the potentials kept.

    ``None`` stays ``None`` (a cold start).  A dataclass with an
    ``eigensolver_state`` field (``SCFWarmStart``, ``SpinorWarmStart``) gets
    ``None`` there (``(None, None)`` for a pair of spin channels) and, where it
    has one, ``first_iteration_degree = None`` (the degree of the input,
    ``polym = polym_t`` of ``parsec.F90:2955``); the eigensolvers then start
    from new vectors (the accelerated ones reset the subspace they keep on
    the device for a ``None`` state).  Bare potentials (a tuple, list or array:
    the spin k-point loop of ``fortran_periodic_backend``) carry no subspace and
    come back unchanged.  Any other type is refused: it needs the
    ``drop_subspace`` of its backend.
    """

    if carried is None:
        return None
    if is_dataclass(carried) and not isinstance(carried, type):
        names = {item.name for item in fields(carried)}
        if "eigensolver_state" in names:
            state = carried.eigensolver_state
            changes = {"eigensolver_state": tuple(None for _ in state) if isinstance(state, tuple) else None}
            if "first_iteration_degree" in names:
                changes["first_iteration_degree"] = None
            return replace(carried, **changes)
    if isinstance(carried, (tuple, list, np.ndarray)):
        return carried
    raise TypeError(
        f"Filter_Restart_After cannot drop the subspace of a {type(carried).__name__} warm start; "
        "its ionic backend must supply drop_subspace"
    )


@dataclass
class FilterRestart:
    """The counters of ``Filter_Restart_After`` over one ionic run."""

    after: int
    restarts_left: int
    densities_left: int

    @classmethod
    def from_eigensolver(cls, settings) -> "FilterRestart | None":
        """The counters of ``settings.filter_restart``, ``None`` without one."""

        value = getattr(settings, "filter_restart", None)
        if value is None:
            return None
        after, restarts, densities = (int(item) for item in value)
        return cls(after=after, restarts_left=restarts, densities_left=densities)

    def warm_start_after_move(self, carried, movements: int, drop_subspace=None):
        """The warm start of the structure after ``movements`` movements, and what was reset.

        Returns ``(warm start, None)`` when nothing is restarted, else the new
        warm start (``None`` for a cold start) and ``"density"`` or
        ``"eigensolver"``.  ``drop_subspace`` is the backend's own way of
        dropping the subspace of its warm start (``IonicBackend.drop_subspace``),
        :func:`drop_eigensolver_subspace` where it has none.
        """

        if self.after < 1 or movements % self.after != 0 or self.restarts_left <= 0:
            return carried, None
        self.restarts_left -= 1
        if self.densities_left > 0:
            self.densities_left -= 1
            return None, "density"
        if carried is None:
            return None, "eigensolver"
        drop = drop_eigensolver_subspace if drop_subspace is None else drop_subspace
        return drop(carried), "eigensolver"


__all__ = ["FilterRestart", "drop_eigensolver_subspace"]
