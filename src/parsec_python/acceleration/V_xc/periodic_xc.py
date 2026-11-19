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
* PBE of ``evaluate_xc_spin_polarized`` (the k-point and Gamma spin drivers,
  ``pbe_spin_polarized``: the variables ``s_ab = D n_a . D n_b``, complex-step
  partials) by default since 2026-11-17: its formula on the device with forward
  derivatives (the ``"ab"`` form of :class:`~parsec_python.acceleration.V_xc.device_xc.DeviceSpinXCEvaluator`,
  :mod:`.dual`), the host's numbers to round-off (diamond, 6 x 6 x 6 cubes
  spin-polarized: 37 of the 81 s of SCF on 16 A100 were this functional on the
  host of every rank);
* PBE of :meth:`~parsec_python.SCF.pbc.PeriodicPreparedSinglePointSystem.evaluate_spin_xc`
  only with ``PARSEC_CUPY_XC=on``: the device kernel is ``pbe_spin`` (analytic
  partials), the readable form of that method; the two readable forms differ
  by about 5e-10 Ry in the potential (bcc Fe).  The accelerated driver puts PBE
  on the device by default since 2026-10-30 (its host form is ``pbe_spin`` as
  well); ``evaluate_spin_xc`` of these drivers does not follow that default
  (:data:`HOST_UNLESS_ASKED`);
* CA-LSD by default since 2026-11-18 (``PARSEC_PERIODIC_CA_XC=auto``, as the
  unpolarized CA-LDA below): ``ca_lda_spin`` of ``evaluate_spin_xc`` and
  ``ca_lda_spin_polarized`` of ``evaluate_xc_spin_polarized``, each in its own
  operations on the device (the ``"spin_polarized"`` form of
  :class:`~parsec_python.acceleration.V_xc.device_xc.DeviceSpinXCEvaluator`),
  the host's numbers to round-off; ``PARSEC_PERIODIC_CA_XC=host`` keeps the
  host, as before.

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
keeps the readable functionals (its numbers unchanged), except CA-LDA, the
default functional: since 2026-11-18 the device evaluates it in one process
and under ``--mpi`` alike (``PARSEC_PERIODIC_CA_XC=auto``; it took the host
over the full grid at every step before, which ``host`` restores).
"""

from __future__ import annotations

from collections import OrderedDict
import threading

from parsec_python.SCF.pbc import set_periodic_spin_xc_provider, set_periodic_xc_provider
from parsec_python.V_xc.selection import as_spin_polarized_result

from .device_xc import (
    DEVICE_XC_FUNCTIONALS,
    PERIODIC_ONLY_XC_FUNCTIONALS,
    DeviceSpinXCEvaluator,
    DeviceXCEvaluator,
    device_grid_supported,
    device_xc_mode,
    device_xc_selected,
    periodic_device_xc_selected,
)

_CAPACITY = 4

#: Spin functionals that these drivers keep on the host unless
#: ``PARSEC_CUPY_XC=on``, whatever the default of the accelerated driver:
#: the device kernel of PBE is not the readable form of their
#: ``evaluate_xc_spin_polarized`` (see the module docstring).
HOST_UNLESS_ASKED = frozenset({"pbe"})


def periodic_spin_device_selected(functional: str, *, mode: str | None = None, kind: str = "spin") -> bool:
    """Whether the periodic spin drivers evaluate ``functional`` on the device under ``mode``.

    :func:`~parsec_python.acceleration.V_xc.device_xc.device_xc_selected`
    for the spin form, except that a functional of :data:`HOST_UNLESS_ASKED`
    needs ``on`` for ``evaluate_spin_xc`` (``kind="spin"``); PBE of
    ``evaluate_xc_spin_polarized`` (``kind="spin_polarized"``) has its own form
    on the device (``"ab"``) and follows ``PARSEC_CUPY_XC`` as the other
    functionals do.
    """

    mode = device_xc_mode() if mode is None else mode
    if functional in HOST_UNLESS_ASKED and mode != "on" and not (kind == "spin_polarized" and functional == "pbe"):
        return False
    # CA-LSD: PARSEC_PERIODIC_CA_XC (both kinds have their form on the device)
    return periodic_device_xc_selected(functional, spin_polarized=True, mode=mode)


def _form(functional: str, kind: str) -> str:
    """The form of the device evaluator: ``"ab"`` for PBE and ``"spin_polarized"`` for CA-LSD of
    ``evaluate_xc_spin_polarized``, ``"pbe_spin"`` (each functional's ``evaluate_spin_xc`` form) otherwise."""

    if kind == "spin_polarized" and functional == "pbe":
        return "ab"
    if kind == "spin_polarized" and functional == "ca":
        return "spin_polarized"
    return "pbe_spin"


class _Evaluators:
    """One device evaluator per (grid, functional, device); the core density of the present system."""

    def __init__(self) -> None:
        self._entries: OrderedDict[tuple, tuple] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, system, functional: str, device_id: int, *, spin: bool = True, form: str = "pbe_spin"):
        grid = system.grid
        key = (id(grid), functional, device_id, float(system.input.scf.ionization_energy_difference), spin, form)
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
            if spin:
                evaluator.form = form
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


def device_spin_evaluator(system, kind: str = "spin"):
    """The device evaluator of ``system``'s spin functional for ``evaluate_spin_xc`` (``kind="spin"``) or
    ``evaluate_xc_spin_polarized`` (``"spin_polarized"``), or ``None`` where the host evaluates."""

    scf = system.input.scf
    if getattr(scf, "noncollinear", False) or getattr(scf, "self_consistent_spin_orbit", False):
        return None
    functional = scf.xc_functional
    if not periodic_spin_device_selected(functional, kind=kind):
        return None
    grid = system.grid
    if not (hasattr(grid, "cell") and device_grid_supported(grid)):
        return None
    from ..Hartree.periodic_device import hartree_device_id

    device_id = hartree_device_id()
    if device_id is None:
        return None
    return _EVALUATORS.get(system, functional, device_id, form=_form(functional, kind))


def device_unpolarized_evaluator(system):
    """The device evaluator of ``system``'s unpolarized functional, or ``None`` (the host evaluates).

    The replicated drivers of an MPI run (``--mpi``) take it for every functional
    :func:`device_xc_selected` puts on the device, on the device of the rank
    (:func:`~parsec_python.acceleration.Hartree.periodic_device.hartree_device_id`); CA-LDA
    (:func:`periodic_device_xc_selected`, ``PARSEC_PERIODIC_CA_XC``) in a run of one process too, on
    the first device of its k-point plan.  Meta-GGAs, hybrids, spinor runs and the grids
    :func:`device_grid_supported` does not take stay on the host, as does every functional where the
    reference drivers have no device.
    """

    from ..Parallel import mpi

    scf = system.input.scf
    if (
        getattr(scf, "is_meta_gga", False)
        or getattr(scf, "is_hybrid", False)
        or getattr(scf, "noncollinear", False)
        or getattr(scf, "self_consistent_spin_orbit", False)
    ):
        return None
    functional = scf.xc_functional
    if functional not in DEVICE_XC_FUNCTIONALS or not periodic_device_xc_selected(functional):
        return None
    if functional not in PERIODIC_ONLY_XC_FUNCTIONALS and not mpi.requested():
        # one process: the readable functionals (CA-LDA aside)
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

    evaluator = device_spin_evaluator(system, kind)
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
