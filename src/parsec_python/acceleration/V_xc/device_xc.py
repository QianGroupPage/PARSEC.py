"""Exchange-correlation functionals evaluated on a CUDA device (CuPy, float64).

The accelerated driver evaluates CA-LDA with the native C++ evaluator
(:mod:`.native_ca_lda`) and every other functional with the readable NumPy
package :mod:`parsec_python.V_xc` on the host, expanding a symmetry-reduced
density to the full grid first.  This module evaluates those other
functionals on the device instead:

==============================  =======================================
``xc_functional``               device evaluation
==============================  =======================================
``pw92``, ``xalpha``,           pointwise, one value per symmetry orbit
``wigner``, ``hedin_lundqvist``
``lb94``, ``casida_salahub``    CA energy, CA potential minus the LB94
                                correction of the density gradient
``pbe``, ``pw91``               GGA energy and the potential of the
                                discrete energy (``pbe.py``)
``ca`` (periodic cells only)    pointwise, one value per symmetry orbit
==============================  =======================================

and the spin-polarized forms of ``pw92``, ``lb94``, ``casida_salahub``,
``pbe``, ``pw91`` and (periodic cells) ``ca`` that
:func:`parsec_python.SCF.run_spin_scf` evaluates (``evaluate_spin_xc``).

The discretization is that of the readable functionals, not a new one.  The
density gradient is ``D n`` of :func:`parsec_python.V_xc.pbe.density_gradient`
(the centered first-derivative stencil of order ``Expansion_Order``, zero
outside a cluster domain, wrapped around a periodic cell), summed in the order
of ``scipy.ndimage.correlate1d``, and the GGA potential is
``df/dn + D.T (2 df/dsigma D n)``, the derivative of the discrete energy
``h**3 sum_i f_i``.  The pointwise formulas of ``device_xc_kernels.cu`` are
line-by-line translations of the NumPy ones, compiled without fused
multiply-adds; the results differ from the host's in the last bits of the
elementary functions (``cbrt``, ``pow``, ``log1p``, ``expm1``, ``exp``,
``asinh``) and of the grid sums of the energy.

With a symmetry reduction (``SymmetrySCFReducer``) the functional is
evaluated once per orbit.  The density of an orbit is the same at all its
points, so its gradient follows from the orbit values at the stencil
neighbours of the representative point; the GGA flux ``2 f_sigma D n`` is a
vector field, and its component along an axis at another point of an orbit is
that of the representative with the sign of the reflection that carries one
onto the other.  No full-grid field is formed, on the device or the host.

``PARSEC_CUPY_XC`` chooses the path of the CuPy backend:

* ``auto`` (the default): the device for every functional above, PBE among
  them (since 2026-10-30; PBE took the host before, and the device changes
  its results by round-off: 6e-14 of the total energy of C795H300);
* ``on``: the same in the accelerated driver; it also moves the spin PBE of
  the reference periodic drivers onto the device, which ``auto`` leaves on
  the host (:mod:`.periodic_xc`);
* ``off``: the host for every functional, as before this module (PBE as up
  to 2026-10-30).

CA-LDA, the default functional, keeps its own path in a cluster (the native
C++ evaluator on the wedge, :mod:`.native_ca_lda`, whatever ``PARSEC_CUPY_XC``
says).  A periodic cell has no native evaluator (its Hartree backend is never
``native``) and evaluated CA-LDA with NumPy on the host over the full grid at
every SCF step, on every periodic route, until 2026-11-18 (about 3.4 s per
step for the 46.7 M points of a diamond cell of 20^3 cubes).  Since then the
device evaluates it (:func:`periodic_device_xc_selected`): on the orbits of
the sectors of a Gamma cell (one value per orbit, as the other LDAs), on the
full grid of the k-point drivers (:mod:`.periodic_xc`), the unpolarized form
``ca_lda`` and both spin forms (``ca_lda_spin`` of ``evaluate_spin_xc``,
``ca_lda_spin_polarized`` of ``evaluate_xc_spin_polarized``), each a
line-by-line translation of the NumPy one, the host's numbers to round-off.
``PARSEC_PERIODIC_CA_XC=host`` (or ``PARSEC_CUPY_XC=off``) keeps the host
functional, as before.

Memory: an evaluator keeps a value map and stencil signs per box point and
a position and multiplicity per value on its device, and takes about a
dozen fields of the values from CuPy's pool while it evaluates
(:attr:`DeviceXCEvaluator.planned_device_bytes`); the accelerated driver
tells the sector eigensolver, whose memory rules count them on that device.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Callable

import numpy as np

from parsec_python.V_xc import XCResult
from parsec_python.V_xc.ca_lda_spin import SpinXCResult
from parsec_python.V_xc.pbe import first_derivative_coefficients
from parsec_python.V_xc.selection import XC_DISPLAY_NAMES

#: Functionals this module evaluates without spin polarization.
DEVICE_XC_FUNCTIONALS = frozenset(
    {
        "pw92",
        "xalpha",
        "wigner",
        "hedin_lundqvist",
        "lb94",
        "casida_salahub",
        "pbe",
        "pw91",
        "ca",
    }
)
#: Functionals this module evaluates with spin polarization.
DEVICE_SPIN_XC_FUNCTIONALS = frozenset(
    {"pw92", "lb94", "casida_salahub", "pbe", "pw91", "ca"}
)
#: Functionals this module evaluates for periodic cells only
#: (:func:`periodic_device_xc_selected`): CA-LDA, whose clusters keep the
#: native C++ evaluator (:func:`device_xc_selected` never selects it).
PERIODIC_ONLY_XC_FUNCTIONALS = frozenset({"ca"})
#: Functionals on the device unless ``PARSEC_CUPY_XC=off``: those added for
#: parity with Fortran PARSEC, which had no accelerated path before, and PBE
#: (since 2026-10-30; the host before).
DEFAULT_DEVICE_XC_FUNCTIONALS = frozenset(
    {
        "pw92",
        "xalpha",
        "wigner",
        "hedin_lundqvist",
        "lb94",
        "casida_salahub",
        "pbe",
        "pw91",
    }
)
_GRADIENT_FUNCTIONALS = frozenset({"lb94", "casida_salahub", "pbe", "pw91"})
_GGA_FUNCTIONALS = frozenset({"pbe", "pw91"})
_LDA_KERNELS = {
    "pw92": "lda_pw92",
    "xalpha": "lda_xalpha",
    "wigner": "lda_wigner",
    "hedin_lundqvist": "lda_hedin_lundqvist",
    "ca": "lda_ca",
}
_SPIN_LDA_KERNELS = {"pw92": "lda_spin_pw92", "ca": "lda_spin_ca"}
_KERNEL_NAMES = (
    "xc_gradient",
    "xc_adjoint",
    "gga_pbe",
    "gga_pw91",
    "spin_gga_pbe",
    "spin_gga_pw91",
    "lda_pw92",
    "lda_xalpha",
    "lda_wigner",
    "lda_hedin_lundqvist",
    "lda_spin_pw92",
    "lda_ca",
    "lda_spin_ca",
    "lb_unpolarized",
    "lb_spin",
)
_THREADS = 256
# The readable functionals refuse a density below this (round-off above it
# is evaluated as it stands).
_NEGATIVE_DENSITY_LIMIT = -1.0e-14


def device_xc_mode() -> str:
    """``PARSEC_CUPY_XC``: ``auto`` (unset or empty), ``on`` or ``off``."""

    raw = os.environ.get("PARSEC_CUPY_XC", "auto").strip().lower()
    if raw in {"", "auto"}:
        return "auto"
    if raw in {"on", "1", "true", "yes", "device"}:
        return "on"
    if raw in {"off", "0", "false", "no", "host"}:
        return "off"
    raise ValueError("PARSEC_CUPY_XC must be auto, on or off")


def device_xc_selected(
    functional: str, *, spin_polarized: bool = False, mode: str | None = None
) -> bool:
    """Whether ``functional`` runs on the device under ``mode``.

    The rule of every grid; CA-LDA (:data:`PERIODIC_ONLY_XC_FUNCTIONALS`) is
    never selected here: a cluster keeps its native evaluator, and a periodic
    cell asks :func:`periodic_device_xc_selected`.
    """

    mode = device_xc_mode() if mode is None else mode
    available = DEVICE_SPIN_XC_FUNCTIONALS if spin_polarized else DEVICE_XC_FUNCTIONALS
    if mode == "off" or functional not in available or functional in PERIODIC_ONLY_XC_FUNCTIONALS:
        return False
    return mode == "on" or functional in DEFAULT_DEVICE_XC_FUNCTIONALS


def periodic_ca_xc_mode() -> str:
    """``PARSEC_PERIODIC_CA_XC``: ``auto`` (unset or empty) or ``host``.

    ``auto``: CA-LDA of a periodic cell on the device wherever the other
    functionals of its route are (:func:`periodic_device_xc_selected`);
    ``host``: the NumPy functional on the full grid, as up to 2026-11-18.
    """

    raw = os.environ.get("PARSEC_PERIODIC_CA_XC", "auto").strip().lower()
    if raw in {"", "auto", "on", "device"}:
        return "auto"
    if raw in {"host", "off", "0", "false", "no"}:
        return "host"
    raise ValueError("PARSEC_PERIODIC_CA_XC must be auto or host")


def periodic_device_xc_selected(
    functional: str, *, spin_polarized: bool = False, mode: str | None = None
) -> bool:
    """Whether a periodic cell evaluates ``functional`` on the device under ``mode`` (``PARSEC_CUPY_XC``).

    :func:`device_xc_selected`, and CA-LDA (both spin forms too) unless
    ``PARSEC_PERIODIC_CA_XC=host`` or ``PARSEC_CUPY_XC=off``.  The grid and
    the device are the caller's to check (:func:`device_grid_supported`).
    """

    mode = device_xc_mode() if mode is None else mode
    if functional in PERIODIC_ONLY_XC_FUNCTIONALS:
        return mode != "off" and periodic_ca_xc_mode() == "auto"
    return device_xc_selected(functional, spin_polarized=spin_polarized, mode=mode)


def device_grid_supported(grid) -> bool:
    """A cluster grid with its lookup box, or an orthorhombic periodic cell.

    The non-orthogonal grid (``GeneralPeriodicGrid``) and the wire and slab
    grids keep the host functionals.
    """

    if getattr(grid, "is_general_periodic", False) or getattr(
        grid, "is_confined_periodic", False
    ):
        return False
    if _is_periodic(grid) and not getattr(grid, "is_orthorhombic", True) and not _unequal_steps(grid):
        # A grid of Andrew Grieder's routines with oblique steps, or his slab: the gradients carry the metric of the
        # cell (V_xc.pbe._oblique_mix), which these kernels do not.  A cell with its sides along the axes whose steps
        # differ takes a row of weights per axis (_StencilGeometry; PARSEC_PERIODIC_UNEQUAL_STEPS=off: the host).
        return False
    shape = tuple(int(value) for value in getattr(grid, "shape", ()))
    if len(shape) != 3 or int(np.prod(shape)) >= 2**31:
        return False
    if _is_periodic(grid):
        return int(np.prod(shape)) == int(grid.size)
    lookup = getattr(grid, "lookup", None)
    return lookup is not None and tuple(np.shape(lookup)) == shape


def _is_periodic(grid) -> bool:
    # As parsec_python.V_xc.pbe._is_periodic.
    return hasattr(grid, "cell") and hasattr(grid, "points_per_axis")


def _unequal_steps(grid) -> bool:
    """A bulk cell with its sides along the axes whose steps differ, under ``PARSEC_PERIODIC_UNEQUAL_STEPS=auto``
    (:func:`parsec_python.acceleration.Symmetry.axis_reflection.unequal_step_grid`): the kernels take its gradient
    with a step per axis (``V_xc.pbe`` differentiates it with the metric of its diagonal lattice)."""

    if not _is_periodic(grid):
        return False
    from ..Symmetry.axis_reflection import periodic_unequal_steps_setting, unequal_step_grid

    return unequal_step_grid(grid) and periodic_unequal_steps_setting() == "auto"


def _display_name(functional: str) -> str:
    name = XC_DISPLAY_NAMES.get(functional, functional)
    return f"discrete-variational {name}" if functional in _GGA_FUNCTIONALS else name


def _kernel_constants() -> dict[str, float]:
    """The constants of the readable modules, by the names of the CUDA source."""

    from importlib import import_module

    # The package exports functions of the same names as these modules.
    def module(name):
        return import_module(f"parsec_python.V_xc.{name}")

    lb = module("asymptotic_correction")
    ca_lda_spin = module("ca_lda_spin")
    local_density = module("local_density")
    pbe = module("pbe")
    pbe_spin = module("pbe_spin")
    pw91 = module("pw91")
    pw92_lda = module("pw92_lda")

    return {
        "PI": np.pi,
        "DENSITY_THRESHOLD": pbe._DENSITY_THRESHOLD,
        "KF": pbe._KF_COEFFICIENT,
        "PBE_KAPPA": pbe._KAPPA,
        "PBE_MU": pbe._MU,
        "PBE_MU_OVER_KAPPA": pbe._MU / pbe._KAPPA,
        "PBE_BETA": pbe._BETA,
        "PBE_GAMMA": pbe._GAMMA,
        "PBE_BETA_OVER_GAMMA": pbe._BETA / pbe._GAMMA,
        "PBE_CX": pbe._CX,
        "SPIN_GAM": pbe_spin._SPIN_GAM,
        "SPIN_FZZ": pbe_spin._SPIN_FZZ,
        "SPIN_ETA": pbe_spin._SPIN_ETA,
        "PW_AX": pw92_lda._AX,
        "PW_GAM": pw92_lda._GAM,
        "PW_FZZ": pw92_lda._FZZ,
        "PW_A_UNPOLARIZED": pw92_lda._UNPOLARIZED[0],
        "TWO_OVER_PI_A0": pw92_lda._TWO_OVER_PI_A0,
        "THREE_QUARTERS_OVER_PI": 0.75 / np.pi,
        "PW91_A": pw91._A,
        "PW91_A1": pw91._A1,
        "PW91_A2": pw91._A2,
        "PW91_A3": pw91._A3,
        "PW91_A4": pw91._A4,
        "PW91_B1": pw91._B1,
        "PW91_XNU": pw91._XNU,
        "PW91_CC0": pw91._CC0,
        "PW91_CX": pw91._CX,
        "PW91_C1": pw91._C1,
        "PW91_C2": pw91._C2,
        "PW91_C3": pw91._C3,
        "PW91_C4": pw91._C4,
        "PW91_C5": pw91._C5,
        "PW91_C6": pw91._C6,
        "PW91_H1_EXPONENT": pw91._H1_EXPONENT,
        "PW91_KS2_OVER_KF2": pw91._KS2_OVER_KF2,
        "PW91_BET": pw91._BET,
        "PW91_DELT": pw91._DELT,
        "PW91_ETA": pw91._SPIN_ETA,
        "XALPHA_ALPHA": local_density.XALPHA_ALPHA,
        "WIGNER_A": local_density._WIGNER_A,
        "WIGNER_B": local_density._WIGNER_B,
        "HL_INVERSE_A": local_density._HL_INVERSE_A,
        "HL_C": local_density._HL_C,
        # ca_lda.py: np.pi * a0 with a0 = (4/(9 pi))**(1/3).
        "CA_PI_A0": np.pi * (4.0 / (9.0 * np.pi)) ** (1.0 / 3.0),
        "CA_GAMMA": ca_lda_spin._GAMMA,
        "LB_TWO_BETA": 2.0 * lb.LB94_BETA,
        "LB_THREE_BETA": 3.0 * lb.LB94_BETA,
    }


def kernel_source() -> str:
    """The CUDA source with the constants written in."""

    source = Path(__file__).with_name("device_xc_kernels.cu").read_text()
    for name, value in _kernel_constants().items():
        # repr gives the shortest decimal that reads back as the same double.
        source = source.replace(f"@{name}@", f"({float(value)!r})")
    missing = re.findall(r"@[A-Z][A-Z0-9_]*@", source)
    if missing:
        raise RuntimeError(f"constants of the device XC kernels without a value: {missing}")
    return source


_MODULES: dict[tuple, dict[str, object]] = {}


def kernel_names(functional: str, *, spin_polarized: bool) -> tuple[str, ...]:
    """The kernels the evaluator of ``functional`` launches."""

    if functional in _GGA_FUNCTIONALS:
        prefix = "spin_gga_" if spin_polarized else "gga_"
        return ("xc_gradient", "xc_adjoint", prefix + functional)
    if functional in _GRADIENT_FUNCTIONALS:
        return ("xc_gradient", "lb_spin" if spin_polarized else "lb_unpolarized")
    return (_SPIN_LDA_KERNELS[functional],) if spin_polarized else (_LDA_KERNELS[functional],)


def _kernels(cp, device_id: int, names: tuple[str, ...] = _KERNEL_NAMES) -> dict[str, object]:
    """The kernels ``names`` compiled for ``device_id``, once per process.

    A module holds only the kernels asked for: NVRTC takes seconds for all
    of them on a first run, which CuPy's kernel cache then keeps.
    """

    key = (int(device_id), tuple(sorted(names)))
    kernels = _MODULES.get(key)
    if kernels is None:
        options = ("-std=c++11", "--fmad=false") + tuple(
            f"-DKERNEL_{name}" for name in key[1]
        )
        with cp.cuda.Device(device_id):
            module = cp.RawModule(code=kernel_source(), options=options)
            kernels = {name: module.get_function(name) for name in key[1]}
        _MODULES[key] = kernels
    return kernels


def evaluation_fields(functional: str, *, spin_polarized: bool, core: bool = False) -> int:
    """Float64 fields of the values that one evaluation of ``functional`` holds at once.

    Counted from :meth:`DeviceXCEvaluator.__call__` and
    :meth:`DeviceSpinXCEvaluator.__call__`.  Unpolarized: the density, the
    potential, energy density and energy per electron, with the gradient (3)
    for LB94 and Casida-Salahub, and the gradient, ``df/dn`` and the flux
    (3) for PBE and PW91, before the gradient is released.  Spin: both
    densities (and half the core while it is added), both potentials and the
    energy density, with both gradients (6) for LB94/CS, and both gradients,
    both ``df/dn`` and both fluxes (6) for the GGAs.
    """

    if functional in _GGA_FUNCTIONALS:
        fields = 19 if spin_polarized else 11
    elif functional in _GRADIENT_FUNCTIONALS:
        fields = 11 if spin_polarized else 7
    else:
        fields = 5 if spin_polarized else 4
    return fields + (1 if spin_polarized and core else 0)


def _launch(kernel, count: int, *arguments) -> None:
    if count == 0:
        return
    blocks = (int(count) + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,), (np.int64(count),) + tuple(arguments))


def _orbit_flips(grid, reduction) -> np.ndarray:
    """Per grid row, the axes the reflection carrying its representative onto it reverses.

    Bit ``a`` of ``flips[i]`` is set where that operation reverses axis
    ``a``.  Where two operations of the group carry the representative onto
    the same row and differ along an axis, the component of an invariant
    vector field along that axis vanishes at the representative, and either
    sign gives the same value; the identity is taken first.
    """

    from ..Symmetry.axis_reflection import _grid_row_mapping_operation

    rows = np.asarray(reduction.representative_rows, dtype=np.int64)
    full_to_wedge = np.asarray(reduction.full_to_wedge)
    unset = np.uint8(255)
    flips = np.full(reduction.full_size, unset, dtype=np.uint8)
    operations = sorted(
        (tuple(int(value) for value in signs) for signs in np.asarray(reduction.signs)),
        key=lambda signs: sum(value < 0 for value in signs),
    )
    for signs in operations:
        images = _grid_row_mapping_operation(
            grid, np.diag(np.asarray(signs, dtype=np.int8)), 1.0e-10, selected_rows=rows
        )
        if images is None or not np.array_equal(
            full_to_wedge[images], np.arange(rows.size)
        ):
            raise RuntimeError(
                "the symmetry operations do not map the orbit representatives "
                "onto their orbits"
            )
        code = np.uint8(sum(1 << axis for axis in range(3) if signs[axis] < 0))
        images = images[flips[images] == unset]
        flips[images] = code
    if np.any(flips == unset):
        raise RuntimeError("a grid row is the image of no orbit representative")
    return flips


def _orbit_operations(grid, reduction) -> tuple[np.ndarray, np.ndarray]:
    """Per grid row, the operation (a signed permutation) carrying its representative onto it, and their table.

    The counterpart of :func:`_orbit_flips` for a group whose operations permute the axes
    (:class:`..Symmetry.PointGroupReduction`, :class:`..Symmetry.SignedPermutationReduction`): row ``i``
    gets the index of the first operation (the identity first) that maps its representative onto it, and
    ``table[g] = (perm_0, perm_1, perm_2, sign_0, sign_1, sign_2)`` with ``R_g[a, perm_a] = sign_a``, so that
    component ``a`` of the image of a vector is ``sign_a`` times its component ``perm_a``.  Where two
    operations carry the representative onto the same row, an invariant vector field takes the same value
    either way.
    """

    from ..Symmetry.axis_reflection import _grid_row_mapping_operation

    rows = np.asarray(reduction.representative_rows, dtype=np.int64)
    full_to_wedge = np.asarray(reduction.full_to_wedge)
    operations = np.asarray(reduction.operations, dtype=np.int64)
    if operations.shape[0] > 255:
        raise ValueError("too many symmetry operations for a byte per row")
    order = sorted(range(operations.shape[0]), key=lambda g: not np.array_equal(operations[g], np.eye(3)))
    unset = np.uint8(255)
    codes = np.full(reduction.full_size, unset, dtype=np.uint8)
    table = np.zeros((operations.shape[0], 6), dtype=np.int8)
    for g in range(operations.shape[0]):
        permutation = np.argmax(np.abs(operations[g]), axis=1)
        table[g, :3] = permutation
        table[g, 3:] = operations[g][np.arange(3), permutation]
    # r -> R r + t in a cell that repeats a smaller one: the translation moves the row, the vector turns by R alone
    translations = getattr(reduction, "translations", None)
    for g in order:
        images = _grid_row_mapping_operation(
            grid, operations[g].astype(np.int8), 1.0e-10, selected_rows=rows,
            translation=None if translations is None else translations[g],
        )
        if images is None or not np.array_equal(full_to_wedge[images], np.arange(rows.size)):
            raise RuntimeError(
                "the symmetry operations do not map the orbit representatives onto their orbits"
            )
        images = images[codes[images] == unset]
        codes[images] = np.uint8(g)
    if np.any(codes == unset):
        raise RuntimeError("a grid row is the image of no orbit representative")
    return codes, table


def _permutes_axes(reduction) -> bool:
    operations = getattr(reduction, "operations", None)
    if operations is None:
        return False
    operations = np.asarray(operations)
    return bool(np.any(np.abs(operations) != np.eye(3, dtype=operations.dtype)[None]))


def _grid_order_identity(reduction, size: int) -> bool:
    """Whether ``reduction`` is the identity alone with orbit ``w`` the grid row ``w`` (``size`` rows)."""

    if int(reduction.group_order) != 1 or int(reduction.wedge_size) != int(size):
        return False
    rows = np.arange(int(size), dtype=np.int64)
    return bool(
        np.array_equal(np.asarray(reduction.representative_rows), rows)
        and np.array_equal(np.asarray(reduction.full_to_wedge), rows)
    )


class _StencilGeometry:
    """The value map, representative positions and stencil weights on a device."""

    def __init__(self, cp, grid, reduction, *, flips: bool) -> None:
        shape = tuple(int(value) for value in grid.shape)
        box_size = int(np.prod(shape))
        self.shape = shape
        self.periodic = _is_periodic(grid)
        self.width = int(grid.settings.expansion_order) // 2
        weights = first_derivative_coefficients(grid.settings.expansion_order)
        # pbe.py _derivative_stencil: the weights in 1/bohr; the transpose
        # correlates with their negatives.  A periodic cell whose steps differ
        # between the axes (unequal_step_grid): a row per axis, weights / h_a
        # (V_xc.pbe._oblique_mix of a diagonal lattice, the steps applied to
        # every weight instead of the sum), the kernels reading row a at
        # a * axis_stride.
        self.axis_stride = 0
        if _unequal_steps(grid):
            steps = np.diag(np.asarray(grid.step_vectors, dtype=np.float64))
            weights = np.ascontiguousarray(weights[None, :] / steps[:, None])
            self.axis_stride = int(weights.shape[1])
        else:
            weights = weights / grid.spacing
        self.weights = cp.asarray(weights)
        self.negated_weights = cp.asarray(-weights)
        self.flips = None
        # 1: ``flips`` holds axis reversals; 2: operation indices into ``table`` (see device_xc_kernels.cu).
        self.flip_mode = 1
        self.table = cp.zeros(6, dtype=np.int8)
        if self.periodic and (reduction is None or _grid_order_identity(reduction, box_size)):
            # (The identity alone, one orbit per point in the order of the
            # grid, as one sector of the identity has it: the box itself.)
            # pbe.py _to_box: the rows of a periodic cell are its box, row-major.
            positions = np.arange(box_size, dtype=np.int32)
            value_of_box = positions
        elif self.periodic:
            # The orbits of the point group of the cell: its rows are its box,
            # row-major, every point of it active, and the stencil wraps.
            if int(reduction.full_size) != box_size:
                raise ValueError("the symmetry orbits do not cover the periodic grid")
            value_of_box = np.asarray(reduction.full_to_wedge).astype(np.int32)
            positions = np.asarray(reduction.representative_rows).astype(np.int32)
            if flips and reduction.group_order > 1 and getattr(reduction, "operations", None) is not None:
                row_codes, table = _orbit_operations(grid, reduction)
                self.flips = cp.asarray(row_codes)
                self.table = cp.asarray(table.reshape(-1))
                self.flip_mode = 2
            elif flips and reduction.group_order > 1:
                self.flips = cp.asarray(_orbit_flips(grid, reduction))
        else:
            lookup = np.asarray(grid.lookup).reshape(-1)
            local = np.asarray(grid.integer_coordinates, dtype=np.int64) - np.asarray(
                grid.index_min, dtype=np.int64
            )
            row_positions = np.ravel_multi_index(local.T, shape)
            active = lookup >= 0
            if reduction is None:
                value_of_box = lookup.astype(np.int32)
                positions = row_positions.astype(np.int32)
            else:
                value_of_box = np.full(box_size, -1, dtype=np.int32)
                value_of_box[active] = np.asarray(reduction.full_to_wedge)[lookup[active]]
                positions = row_positions[
                    np.asarray(reduction.representative_rows, dtype=np.int64)
                ].astype(np.int32)
                if flips and reduction.group_order > 1 and getattr(reduction, "operations", None) is not None:
                    # A group given by its operations (those that permute the axes, or a subgroup whose
                    # ``signs`` are placeholders): an operation index per point and their table.
                    row_codes, table = _orbit_operations(grid, reduction)
                    box_codes = np.zeros(box_size, dtype=np.uint8)
                    box_codes[active] = row_codes[lookup[active]]
                    self.flips = cp.asarray(box_codes)
                    self.table = cp.asarray(table.reshape(-1))
                    self.flip_mode = 2
                elif flips and reduction.group_order > 1:
                    row_flips = _orbit_flips(grid, reduction)
                    box_flips = np.zeros(box_size, dtype=np.uint8)
                    box_flips[active] = row_flips[lookup[active]]
                    self.flips = cp.asarray(box_flips)
        self.value_of_box = cp.asarray(value_of_box)
        self.positions = cp.asarray(positions)
        self.count = int(positions.size)
        # A placeholder for the flips argument where none are needed.
        self._no_flips = cp.zeros(1, dtype=np.uint8)

    @property
    def device_bytes(self) -> int:
        arrays = (self.value_of_box, self.positions, self.flips)
        return int(sum(array.nbytes for array in arrays if array is not None))

    def _box_arguments(self):
        n0, n1, n2 = self.shape
        return (
            np.int32(n0),
            np.int32(n1),
            np.int32(n2),
            np.int32(1 if self.periodic else 0),
            np.int32(self.width),
            np.int32(self.axis_stride),
        )

    def gradient(self, cp, kernels, values):
        """``D n`` at every value, shape ``(3, count)`` (pbe.py ``density_gradient``)."""

        gradient = cp.empty((3, self.count), dtype=np.float64)
        _launch(
            kernels["xc_gradient"],
            self.count,
            values,
            self.value_of_box,
            self.positions,
            *self._box_arguments(),
            self.weights,
            gradient,
        )
        return gradient

    def adjoint(self, cp, kernels, flux, base, scale: float):
        """``scale * (base + sum_a D_a.T flux_a)`` (pbe.py ``discrete_gga``)."""

        out = cp.empty(self.count, dtype=np.float64)
        flips = self._no_flips if self.flips is None else self.flips
        _launch(
            kernels["xc_adjoint"],
            self.count,
            flux,
            self.value_of_box,
            flips,
            np.int32(0 if self.flips is None else self.flip_mode),
            self.table,
            self.positions,
            *self._box_arguments(),
            self.negated_weights,
            base,
            np.float64(scale),
            out,
        )
        return out


class _DeviceEvaluatorBase:
    """Device, kernels, geometry and the cached frozen core density."""

    spin_polarized = False

    def __init__(
        self,
        functional: str,
        grid,
        *,
        core_density=None,
        reducer=None,
        device_id: int | None = None,
        ionization_energy_difference: float = 0.0,
    ) -> None:
        from ..backends.cupy import require_cupy

        available = (
            DEVICE_SPIN_XC_FUNCTIONALS if self.spin_polarized else DEVICE_XC_FUNCTIONALS
        )
        if functional not in available:
            raise ValueError(f"no device evaluator for the {functional} functional")
        if functional in PERIODIC_ONLY_XC_FUNCTIONALS and not _is_periodic(grid):
            raise ValueError(
                f"the device evaluates the {functional} functional of periodic cells only "
                "(a cluster has the native evaluator)"
            )
        if not device_grid_supported(grid):
            raise ValueError("the device XC evaluators need a cluster or orthorhombic grid")
        cp, _ = require_cupy()
        self._cp = cp
        self.functional = functional
        self.grid = grid
        self.reducer = reducer
        self.volume_element = float(grid.volume_element)
        self.device_id = int(cp.cuda.Device().id if device_id is None else device_id)
        self.cap = (
            float(ionization_energy_difference)
            if functional == "casida_salahub"
            else None
        )
        # A callable reads the core density of the present structure (the
        # driver replaces the prepared system when atoms move).
        self._core_source = (
            core_density if callable(core_density) else (lambda: core_density)
        )
        self._core_host = None
        self._core_device = None
        reduction = None if reducer is None else reducer.reduction
        with cp.cuda.Device(self.device_id):
            self._kernels = _kernels(
                cp,
                self.device_id,
                kernel_names(functional, spin_polarized=self.spin_polarized),
            )
            self.geometry = (
                _StencilGeometry(
                    cp, grid, reduction, flips=functional in _GGA_FUNCTIONALS
                )
                if functional in _GRADIENT_FUNCTIONALS
                else None
            )
            self._multiplicities = (
                None
                if reducer is None
                else cp.asarray(np.asarray(reduction.multiplicities, dtype=np.float64))
            )
        self.size = int(
            grid.size if reducer is None else reducer.reduction.wedge_size
        )
        self.implementation_label = (
            f"CuPy float64 {'spin-polarized ' if self.spin_polarized else ''}"
            f"{_display_name(functional)}"
            + (" on the symmetry orbits" if reducer is not None else "")
        )

    @property
    def device_bytes(self) -> int:
        """Bytes the evaluator keeps on its device between evaluations."""

        geometry = 0 if self.geometry is None else self.geometry.device_bytes
        core = 0 if self._core_device is None else int(self._core_device.nbytes)
        weights = 0 if self._multiplicities is None else int(self._multiplicities.nbytes)
        return geometry + core + weights

    @property
    def planned_device_bytes(self) -> int:
        """The most bytes the evaluator holds on its device, for a memory rule to count.

        What it keeps (:attr:`device_bytes`, with the frozen core density
        counted before its first upload) and the float64 fields of the
        values that one evaluation holds at once (:func:`evaluation_fields`;
        CuPy's pool keeps them for the next).  From the shapes alone: it
        reads no state of the device.  For the 677,542 orbits of C795H300
        with PBE (box of 5.4 M points) 35 + 60 MB.
        """

        geometry = 0 if self.geometry is None else self.geometry.device_bytes
        weights = 0 if self._multiplicities is None else int(self._multiplicities.nbytes)
        core = 8 * self.size if self._core_source() is not None else 0
        fields = evaluation_fields(
            self.functional, spin_polarized=self.spin_polarized, core=core > 0
        )
        return geometry + weights + core + 8 * self.size * fields

    def _values(self, field) -> np.ndarray:
        if self.reducer is not None:
            values = self.reducer.wedge_values(field)
        else:
            values = np.asarray(field, dtype=np.float64)
        values = np.ascontiguousarray(values, dtype=np.float64)
        if values.shape != (self.size,):
            raise ValueError("density does not match the XC grid")
        return values

    def _core(self):
        core = self._core_source()
        if core is None:
            self._core_host = None
            self._core_device = None
            return None
        if core is not self._core_host:
            values = self._values(core)
            self._core_device = self._cp.asarray(values)
            self._core_host = core
        return self._core_device

    def _integral(self, energy_density) -> float:
        cp = self._cp
        total = (
            cp.sum(energy_density)
            if self._multiplicities is None
            else cp.dot(self._multiplicities, energy_density)
        )
        return float(self.volume_element * float(total))

    def _field(self, values):
        host = self._cp.asnumpy(values)
        return host if self.reducer is None else self.reducer.field(host)

    def _check(self, density, message: str) -> None:
        if density.size and float(self._cp.min(density)) < _NEGATIVE_DENSITY_LIMIT:
            raise ValueError(message)


class DeviceXCEvaluator(_DeviceEvaluatorBase):
    """An unpolarized functional on a device: ``evaluator(density) -> XCResult``.

    ``density`` is the valence density of the SCF loop: a full-grid array,
    or with ``reducer`` a ``SymmetryScalarField`` (or a full-grid array,
    orbit-averaged first, as the native CA evaluator takes it).  The result
    holds the fields of :meth:`PreparedSinglePointSystem.evaluate_xc`, as
    wedge fields under ``reducer``; the frozen core density is added as there.
    """

    def __call__(self, valence_density) -> XCResult:
        cp = self._cp
        functional = self.functional
        name = XC_DISPLAY_NAMES.get(functional, functional)
        with cp.cuda.Device(self.device_id):
            density = cp.asarray(self._values(valence_density))
            core = self._core()
            if core is not None:
                density = density + core
            self._check(density, f"{name} requires a nonnegative density")
            count = self.size
            potential = cp.empty(count, dtype=np.float64)
            energy_density = cp.empty(count, dtype=np.float64)
            epsilon = cp.empty(count, dtype=np.float64)
            kernels = self._kernels
            if functional in _LDA_KERNELS:
                _launch(
                    kernels[_LDA_KERNELS[functional]],
                    count,
                    density,
                    potential,
                    energy_density,
                    epsilon,
                )
            elif functional in _GGA_FUNCTIONALS:
                gradient = self.geometry.gradient(cp, kernels, density)
                f_n = cp.empty(count, dtype=np.float64)
                flux = cp.empty((3, count), dtype=np.float64)
                _launch(
                    kernels["gga_pbe" if functional == "pbe" else "gga_pw91"],
                    count,
                    density,
                    gradient,
                    energy_density,
                    f_n,
                    flux,
                    epsilon,
                )
                del gradient
                # Hartree -> Rydberg at the public boundary, as pbe.py does.
                potential = self.geometry.adjoint(cp, kernels, flux, f_n, 2.0)
            else:
                gradient = self.geometry.gradient(cp, kernels, density)
                _launch(
                    kernels["lb_unpolarized"],
                    count,
                    density,
                    gradient,
                    # asymptotic_correction.py: x = 2**(1/3)|grad rho|/rho**(4/3).
                    np.float64(2.0 ** (1.0 / 3.0)),
                    np.int32(self.cap is not None),
                    np.float64(0.0 if self.cap is None else self.cap),
                    potential,
                    energy_density,
                    epsilon,
                )
            total_energy = self._integral(energy_density)
            return XCResult(
                potential=self._field(potential),
                energy_per_electron=self._field(epsilon),
                energy_density=self._field(energy_density),
                total_energy=total_energy,
            )


class DeviceSpinXCEvaluator(_DeviceEvaluatorBase):
    """A spin-polarized functional on a device: ``evaluator(up, down) -> SpinXCResult``.

    The counterpart of :meth:`PreparedSinglePointSystem.evaluate_spin_xc`
    (each channel sees half of the frozen core density): full-grid channels,
    or with ``reducer`` the orbit fields of a spin loop in the symmetry
    sectors (``Spin_Symmetry_Sectors``), whose results are orbit fields.  Both
    channel densities of the sectors are invariant, so the orbit geometry of
    the unpolarized functional applies to each: the gradient of a channel
    from its orbit values, ``D n_a . D n_b`` invariant, and each channel's
    flux a vector field carried by the reflection signs.
    """

    spin_polarized = True
    #: The form of PBE: ``"pbe_spin"``, the kernel of :func:`parsec_python.V_xc.pbe_spin.pbe_spin`
    #: (``evaluate_spin_xc``), or ``"ab"``, :func:`parsec_python.V_xc.pbe.pbe_spin_polarized` in the variables
    #: ``s_ab = D n_a . D n_b`` (``evaluate_xc_spin_polarized`` of the periodic spin drivers): its formula on the
    #: device with forward derivatives in place of the complex step (:mod:`.dual`), the same partials to round-off.
    #: The form of CA-LDA: :func:`parsec_python.V_xc.ca_lda_spin.ca_lda_spin` (``evaluate_spin_xc``) unless
    #: ``"spin_polarized"``, :func:`parsec_python.V_xc.ca_lda.ca_lda_spin_polarized` (``evaluate_xc_spin_polarized``
    #: of the periodic spin drivers: the same formula in other operations, and no refusal of a negative channel).
    form = "pbe_spin"
    #: points of one pass of the ``"ab"`` form (its temporaries are a few dozen fields of this length)
    ab_chunk = 1 << 20

    def __call__(self, valence_up, valence_down) -> SpinXCResult:
        if self.functional == "pbe" and self.form == "ab" and self.reducer is None:
            return self._pbe_ab(valence_up, valence_down)
        cp = self._cp
        functional = self.functional
        name = XC_DISPLAY_NAMES.get(functional, functional)
        with cp.cuda.Device(self.device_id):
            up = cp.asarray(self._values(valence_up))
            down = cp.asarray(self._values(valence_down))
            core = self._core()
            if core is not None:
                # ca_lda_spin.py spin_densities_with_core.
                half = 0.5 * core
                up = up + half
                down = down + half
            polarized_ca = functional == "ca" and self.form == "spin_polarized"
            if not polarized_ca:
                # (ca_lda_spin_polarized refuses no density: it takes a channel below zero as empty.)
                message = f"{name} requires nonnegative spin densities"
                self._check(up, message)
                self._check(down, message)
            up = cp.maximum(up, 0.0)
            down = cp.maximum(down, 0.0)
            count = self.size
            potential_up = cp.empty(count, dtype=np.float64)
            potential_down = cp.empty(count, dtype=np.float64)
            energy_density = cp.empty(count, dtype=np.float64)
            kernels = self._kernels
            if functional == "ca":
                _launch(
                    kernels["lda_spin_ca"],
                    count,
                    up,
                    down,
                    np.int32(1 if polarized_ca else 0),
                    potential_up,
                    potential_down,
                    energy_density,
                )
            elif functional == "pw92":
                _launch(
                    kernels["lda_spin_pw92"],
                    count,
                    up,
                    down,
                    potential_up,
                    potential_down,
                    energy_density,
                )
            elif functional in _GGA_FUNCTIONALS:
                gradient_up = self.geometry.gradient(cp, kernels, up)
                gradient_down = self.geometry.gradient(cp, kernels, down)
                f_up = cp.empty(count, dtype=np.float64)
                f_down = cp.empty(count, dtype=np.float64)
                flux_up = cp.empty((3, count), dtype=np.float64)
                flux_down = cp.empty((3, count), dtype=np.float64)
                _launch(
                    kernels["spin_gga_pbe" if functional == "pbe" else "spin_gga_pw91"],
                    count,
                    up,
                    down,
                    gradient_up,
                    gradient_down,
                    energy_density,
                    f_up,
                    f_down,
                    flux_up,
                    flux_down,
                )
                del gradient_up, gradient_down
                potential_up = self.geometry.adjoint(cp, kernels, flux_up, f_up, 2.0)
                potential_down = self.geometry.adjoint(cp, kernels, flux_down, f_down, 2.0)
            else:
                gradient_up = self.geometry.gradient(cp, kernels, up)
                gradient_down = self.geometry.gradient(cp, kernels, down)
                _launch(
                    kernels["lb_spin"],
                    count,
                    up,
                    down,
                    gradient_up,
                    gradient_down,
                    np.int32(self.cap is not None),
                    np.float64(0.0 if self.cap is None else self.cap),
                    potential_up,
                    potential_down,
                    energy_density,
                )
            total_energy = self._integral(energy_density)
            return SpinXCResult(
                potential_up=self._field(potential_up),
                potential_down=self._field(potential_down),
                energy_density=self._field(energy_density),
                total_energy=total_energy,
            )


    def _pbe_ab(self, valence_up, valence_down) -> SpinXCResult:
        """:func:`parsec_python.V_xc.pbe.pbe_spin_polarized` on the device (the ``"ab"`` form).

        The steps of :func:`~parsec_python.V_xc.pbe.discrete_spin_gga_ab`: the gradients of the two channels
        (each with half the core density), ``s_ab``, the energy and its five partials at the points with density
        (:func:`~parsec_python.V_xc.pbe.pbe_spin_polarized_partials`: the same floors; the points of an empty
        channel take its limit, on the host, as there), the fluxes ``2 f_aa D n_a + f_ab D n_b`` and the exact
        transpose of the gradient.
        """

        from parsec_python.V_xc.pbe import (
            _DENSITY_THRESHOLD,
            _SPIN_DENSITY_FLOOR,
            _pbe_spin_energy_density,
        )
        from parsec_python.V_xc.pbe_spin import pbe_spin_energy_partials

        from .dual import partials

        cp = self._cp
        with cp.cuda.Device(self.device_id):
            up = cp.asarray(self._values(valence_up))
            down = cp.asarray(self._values(valence_down))
            core = self._core()
            if core is not None:
                half = 0.5 * core
                up = up + half
                down = down + half
            message = "PBE requires nonnegative spin densities"
            self._check(up, message)
            self._check(down, message)
            kernels = self._kernels
            gradient_up = self.geometry.gradient(cp, kernels, up)
            gradient_down = self.geometry.gradient(cp, kernels, down)
            s_uu = gradient_up[0] * gradient_up[0] + gradient_up[1] * gradient_up[1] + gradient_up[2] * gradient_up[2]
            s_dd = (
                gradient_down[0] * gradient_down[0]
                + gradient_down[1] * gradient_down[1]
                + gradient_down[2] * gradient_down[2]
            )
            s_ud = gradient_up[0] * gradient_down[0] + gradient_up[1] * gradient_down[1] + gradient_up[2] * gradient_down[2]
            outputs = [cp.zeros(self.size, dtype=np.float64) for _ in range(6)]
            total = up + down
            active = cp.flatnonzero(total > _DENSITY_THRESHOLD)
            for start in range(0, int(active.size), int(self.ab_chunk)):
                rows = active[start : start + int(self.ab_chunk)]
                found = partials(
                    _pbe_spin_energy_density,
                    (
                        cp.maximum(up[rows], _SPIN_DENSITY_FLOOR),
                        cp.maximum(down[rows], _SPIN_DENSITY_FLOOR),
                        cp.maximum(s_uu[rows], 0.0),
                        s_ud[rows],
                        cp.maximum(s_dd[rows], 0.0),
                    ),
                    cp,
                )
                for output, value in zip(outputs, found):
                    output[rows] = value
                del found
            empty = cp.flatnonzero((total > _DENSITY_THRESHOLD) & ((up <= 0.0) | (down <= 0.0)))
            del total, active
            if int(empty.size):
                # the fully polarized limit with the regularized phi of pbe_spin (pbe_spin_polarized_partials)
                host = [cp.asnumpy(field[empty]) for field in (up, down, s_uu, s_ud, s_dd)]
                suu, sud, sdd = np.maximum(host[2], 0.0), host[3], np.maximum(host[4], 0.0)
                f, d_up, d_down, d_sigma_up, d_sigma_down, d_sigma_total = pbe_spin_energy_partials(
                    np.maximum(host[0], 0.0), np.maximum(host[1], 0.0), suu, sdd, np.maximum(suu + 2.0 * sud + sdd, 0.0)
                )
                for output, value in zip(
                    outputs,
                    (f, d_up, d_down, d_sigma_up + d_sigma_total, 2.0 * d_sigma_total, d_sigma_down + d_sigma_total),
                ):
                    output[empty] = cp.asarray(value)
            f, f_up, f_down, f_uu, f_ud, f_dd = outputs
            del s_uu, s_ud, s_dd
            flux_up = 2.0 * f_uu * gradient_up + f_ud * gradient_down
            flux_down = 2.0 * f_dd * gradient_down + f_ud * gradient_up
            del gradient_up, gradient_down
            potential_up = self.geometry.adjoint(cp, kernels, flux_up, f_up, 2.0)
            potential_down = self.geometry.adjoint(cp, kernels, flux_down, f_down, 2.0)
            energy_density = 2.0 * f
            return SpinXCResult(
                potential_up=self._field(potential_up),
                potential_down=self._field(potential_down),
                energy_density=self._field(energy_density),
                total_energy=self._integral(energy_density),
            )


def accelerated_xc_evaluators(
    reference,
    *,
    backend: str,
    reducer=None,
    device_id: int | None = None,
    core_density: Callable[[], np.ndarray | None] | None = None,
):
    """The device evaluators the accelerated driver uses, ``(unpolarized, spin)``.

    Each is ``None`` where the host keeps the functional: another backend
    than CuPy, CA-LDA of a cluster (the driver's native evaluator), a
    functional ``PARSEC_CUPY_XC`` (or, for CA-LDA of a periodic cell,
    ``PARSEC_PERIODIC_CA_XC``) leaves on the host, or a grid this module does
    not handle.  ``core_density`` returns the frozen core density of the
    present structure (default: that of ``reference``).
    """

    scf = getattr(getattr(reference, "input", None), "scf", None)
    functional = getattr(scf, "xc_functional", "ca")
    if backend != "cupy" or scf is None:
        return None, None
    grid = reference.grid
    if not device_grid_supported(grid):
        return None, None
    spin = bool(getattr(scf, "spin_polarized", False))
    keywords = {
        "core_density": (
            core_density if core_density is not None else (lambda: reference.core_density)
        ),
        "device_id": device_id,
        "ionization_energy_difference": float(
            getattr(scf, "ionization_energy_difference", 0.0)
        ),
    }
    # Zero_Average_Vxc of a 3-D cell (potshift): the readable evaluate_xc
    # shifts its potentials; the device evaluators get the same shift.
    shift = getattr(reference, "_zero_average", None)
    zero_average = callable(shift) and shift(None) is True
    from parsec_python.V_xc.libxc import is_libxc_functional

    if is_libxc_functional(functional):
        # A libxc LDA/GGA on the device where libxc is a CUDA build (device_libxc.py);
        # the host evaluates it otherwise, and PARSEC_CUPY_XC=off keeps it there.
        if device_xc_mode() == "off" or getattr(scf, "is_hybrid", False):
            return None, None
        from .device_libxc import libxc_xc_evaluators

        unpolarized, spin_evaluator = libxc_xc_evaluators(
            reference,
            spec=functional,
            reducer=reducer,
            device_id=device_id,
            core_density=keywords["core_density"],
        )
        if zero_average:
            unpolarized = None if unpolarized is None else ZeroAverageXC(unpolarized)
            spin_evaluator = None if spin_evaluator is None else ZeroAverageXC(spin_evaluator)
        return unpolarized, spin_evaluator
    # A periodic cell takes CA-LDA too (it has no native evaluator).
    selected = periodic_device_xc_selected if _is_periodic(grid) else device_xc_selected
    if spin:
        # The loop of two channels evaluates the spin functional only; on the
        # orbits where the channels are solved in the symmetry sectors.
        if not selected(functional, spin_polarized=True):
            return None, None
        evaluator = DeviceSpinXCEvaluator(functional, grid, reducer=reducer, **keywords)
        return None, (ZeroAverageXC(evaluator) if zero_average else evaluator)
    if not selected(functional):
        return None, None
    evaluator = DeviceXCEvaluator(functional, grid, reducer=reducer, **keywords)
    return (ZeroAverageXC(evaluator) if zero_average else evaluator), None


class ZeroAverageXC:
    """A device evaluator whose potentials lose their average (``potshift.f90``).

    What :meth:`parsec_python.SCF.pbc.PeriodicPreparedSinglePointSystem.evaluate_xc`
    does with ``Zero_Average_Vxc``: every channel's potential minus its plain
    mean over the full periodic grid (``sum V hcub / vcell``); energies
    unchanged.  The evaluator's other attributes are its own.
    """

    def __init__(self, evaluator) -> None:
        self.evaluator = evaluator
        self.implementation_label = (
            f"{evaluator.implementation_label}, zero average (Zero_Average_Vxc)"
        )

    def __getattr__(self, name: str):
        return getattr(self.evaluator, name)

    @staticmethod
    def _shifted(potential):
        reduction = getattr(potential, "reduction", None)
        if reduction is not None:
            # One value per orbit (an evaluator with a reducer): the mean over
            # the full grid weighs every orbit by its size.
            weights = np.asarray(reduction.multiplicities, dtype=np.float64)
            mean = float(np.dot(weights, potential.values) / np.sum(weights))
            return type(potential)(reduction, potential.values - mean)
        values = np.asarray(potential, dtype=np.float64)
        return values - float(np.mean(values))

    def __call__(self, *densities):
        from dataclasses import fields, replace

        result = self.evaluator(*densities)
        names = {field.name for field in fields(result)}
        changed = {
            name: self._shifted(getattr(result, name))
            for name in ("potential", "potential_up", "potential_down")
            if name in names
        }
        return replace(result, **changed)


__all__ = [
    "DEFAULT_DEVICE_XC_FUNCTIONALS",
    "DEVICE_SPIN_XC_FUNCTIONALS",
    "DEVICE_XC_FUNCTIONALS",
    "DeviceSpinXCEvaluator",
    "DeviceXCEvaluator",
    "ZeroAverageXC",
    "accelerated_xc_evaluators",
    "device_grid_supported",
    "device_xc_mode",
    "device_xc_selected",
    "evaluation_fields",
    "kernel_source",
    "PERIODIC_ONLY_XC_FUNCTIONALS",
    "periodic_ca_xc_mode",
    "periodic_device_xc_selected",
]
