"""The spin functionals of a periodic cell on a CUDA device, for the reference periodic drivers.

The k-point spin driver (:func:`parsec_python.SCF.run_scf_kpoints_spin_polarized`),
the Gamma-point spin driver of a periodic cell and the spin loop of its forces
and ionic steps (:func:`parsec_python.SCF.run_spin_scf`) evaluate the spin
functionals on the host, on the full periodic grid.  Where those drivers put
their eigenproblems on a device (the k-point plan of
:func:`parsec_python.acceleration.Hartree.periodic_device.hartree_device_id`),
this provider evaluates on the device, with
:class:`~parsec_python.acceleration.V_xc.device_xc.DeviceSpinXCEvaluator`
(the parity round's kernels, translations of the readable spin functionals),
the functionals that :func:`periodic_spin_device_selected` names:

* by default (``PARSEC_CUPY_XC=auto``) the functionals added for Fortran
  parity (PW92, PW91, LB94, Casida-Salahub): their ``evaluate_spin_xc``
  semantics are those of the readable path of both methods, so the results
  equal the host ones to round-off;
* PBE only with ``PARSEC_CUPY_XC=on``: the device kernel is ``pbe_spin``
  (analytic partials), which :meth:`~parsec_python.SCF.pbc.PeriodicPreparedSinglePointSystem.evaluate_spin_xc`
  uses, while ``evaluate_xc_spin_polarized`` (the k-point and Gamma spin
  drivers) uses ``pbe_spin_polarized`` (complex-step partials); the two readable
  forms differ by about 5e-10 Ry in the potential (bcc Fe), so the opt-in moves
  those drivers' PBE numbers by that much.  The accelerated driver puts PBE on
  the device by default since 2026-10-30 (its host form is ``pbe_spin`` as
  well, so the device changes it by round-off only); these drivers do not
  follow that default (:data:`HOST_UNLESS_ASKED`);
* CA-LSD stays on the host (as in the accelerated driver).

Not taken: the orthorhombic grid only (Fortran's general grid and the
wire/slab grids keep the host, :func:`~parsec_python.acceleration.V_xc.device_xc.device_grid_supported`),
and neither spin-orbit nor non-collinear runs (their drivers keep the host).
The zero-average shift (``Zero_Average_Vxc``) is applied by the caller as on
the host.  Importing :mod:`parsec_python.acceleration` installs it.

In an MPI run (``--mpi``) the replicated drivers of every rank also evaluate
the unpolarized semilocal functionals on the device of the rank
(:func:`device_unpolarized_evaluator`, :class:`~parsec_python.acceleration.V_xc.device_xc.DeviceXCEvaluator`,
the functionals ``PARSEC_CUPY_XC`` puts on the device, PBE among them): the
kernels translate the readable functionals, so the results differ from the
host ones by round-off (6e-14 of the energy of C795H300), and their
fixed-order sums give every rank the same potential.  A run of one process
keeps the readable functionals (its numbers unchanged).
"""

from __future__ import annotations

from collections import OrderedDict
import threading

from parsec_python.SCF.pbc import set_periodic_spin_xc_provider, set_periodic_xc_provider
from parsec_python.V_xc.selection import as_spin_polarized_result

from .device_xc import (
    DEVICE_XC_FUNCTIONALS,
    DeviceSpinXCEvaluator,
    DeviceXCEvaluator,
    device_grid_supported,
    device_xc_mode,
    device_xc_selected,
)

_CAPACITY = 4

#: Spin functionals that these drivers keep on the host unless
#: ``PARSEC_CUPY_XC=on``, whatever the default of the accelerated driver:
#: the device kernel of PBE is not the readable form of their
#: ``evaluate_xc_spin_polarized`` (see the module docstring).
HOST_UNLESS_ASKED = frozenset({"pbe"})


def periodic_spin_device_selected(functional: str, *, mode: str | None = None) -> bool:
    """Whether the periodic spin drivers evaluate ``functional`` on the device under ``mode``.

    :func:`~parsec_python.acceleration.V_xc.device_xc.device_xc_selected`
    for the spin form, except that a functional of :data:`HOST_UNLESS_ASKED`
    needs ``on``.
    """

    mode = device_xc_mode() if mode is None else mode
    if functional in HOST_UNLESS_ASKED and mode != "on":
        return False
    return device_xc_selected(functional, spin_polarized=True, mode=mode)


class _Evaluators:
    """One device evaluator per (grid, functional, device); the core density of the present system."""

    def __init__(self) -> None:
        self._entries: OrderedDict[tuple, tuple] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, system, functional: str, device_id: int, *, spin: bool = True):
        grid = system.grid
        key = (id(grid), functional, device_id, float(system.input.scf.ionization_energy_difference), spin)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
        if entry is None:
            holder = [system.core_density]
            evaluator = (DeviceSpinXCEvaluator if spin else DeviceXCEvaluator)(
                functional,
                grid,
                core_density=lambda: holder[0],
                device_id=device_id,
                ionization_energy_difference=float(system.input.scf.ionization_energy_difference),
            )
            entry = (grid, holder, evaluator)
            with self._lock:
                self._entries[key] = entry
                while len(self._entries) > _CAPACITY:
                    self._entries.popitem(last=False)
        _, holder, evaluator = entry
        # a moved structure has a new core density (the evaluator uploads it once)
        holder[0] = system.core_density
        return evaluator

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_EVALUATORS = _Evaluators()


def clear_cache() -> None:
    """Release every cached device evaluator."""

    _EVALUATORS.clear()


def device_spin_evaluator(system):
    """The device evaluator of ``system``'s spin functional, or ``None`` where the host evaluates."""

    scf = system.input.scf
    if getattr(scf, "noncollinear", False) or getattr(scf, "self_consistent_spin_orbit", False):
        return None
    functional = scf.xc_functional
    if not periodic_spin_device_selected(functional):
        return None
    grid = system.grid
    if not (hasattr(grid, "cell") and device_grid_supported(grid)):
        return None
    from ..Hartree.periodic_device import hartree_device_id

    device_id = hartree_device_id()
    if device_id is None:
        return None
    return _EVALUATORS.get(system, functional, device_id)


def device_unpolarized_evaluator(system):
    """The device evaluator of ``system``'s unpolarized functional in an MPI run, or ``None`` (the host evaluates).

    Only the replicated drivers of an MPI run (``--mpi``) take it, on the device of the rank
    (:func:`~parsec_python.acceleration.Hartree.periodic_device.hartree_device_id`); meta-GGAs, hybrids,
    spinor runs, CA-LDA (its own path) and the grids :func:`device_grid_supported` does not take stay
    on the host.
    """

    from ..Parallel import mpi

    if not mpi.requested():
        return None
    scf = system.input.scf
    if (
        getattr(scf, "is_meta_gga", False)
        or getattr(scf, "is_hybrid", False)
        or getattr(scf, "noncollinear", False)
        or getattr(scf, "self_consistent_spin_orbit", False)
    ):
        return None
    functional = scf.xc_functional
    if functional not in DEVICE_XC_FUNCTIONALS or not device_xc_selected(functional):
        return None
    grid = system.grid
    if not (hasattr(grid, "cell") and device_grid_supported(grid)):
        return None
    from ..Hartree.periodic_device import hartree_device_id

    device_id = hartree_device_id()
    if device_id is None:
        return None
    return _EVALUATORS.get(system, functional, device_id, spin=False)


def provide_xc(system, density):
    """The hook of :mod:`parsec_python.SCF.pbc` (``set_periodic_xc_provider``)."""

    evaluator = device_unpolarized_evaluator(system)
    if evaluator is None:
        return None
    return evaluator(density)


def provide_spin_xc(system, kind: str, density_up, density_down):
    """The hook of :mod:`parsec_python.SCF.pbc` (``set_periodic_spin_xc_provider``)."""

    evaluator = device_spin_evaluator(system)
    if evaluator is None:
        return None
    result = evaluator(density_up, density_down)
    if kind == "spin_polarized":
        return as_spin_polarized_result(result)
    return result


def install() -> None:
    set_periodic_spin_xc_provider(provide_spin_xc)
    set_periodic_xc_provider(provide_xc)


def uninstall() -> None:
    set_periodic_spin_xc_provider(None)
    set_periodic_xc_provider(None)


__all__ = [
    "HOST_UNLESS_ASKED",
    "clear_cache",
    "device_spin_evaluator",
    "device_unpolarized_evaluator",
    "install",
    "periodic_spin_device_selected",
    "provide_spin_xc",
    "provide_xc",
    "uninstall",
]
