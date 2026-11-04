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
==============================  =======================================

and the spin-polarized forms of ``pw92``, ``lb94``, ``casida_salahub``,
``pbe`` and ``pw91`` that :func:`parsec_python.SCF.run_spin_scf` evaluates
(``evaluate_spin_xc``).

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

CA-LDA keeps its own path in every case.

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
    }
)
#: Functionals this module evaluates with spin polarization.
DEVICE_SPIN_XC_FUNCTIONALS = frozenset(
    {"pw92", "lb94", "casida_salahub", "pbe", "pw91"}
)
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
}
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
    """Whether ``functional`` runs on the device under ``mode``."""

    mode = device_xc_mode() if mode is None else mode
    available = DEVICE_SPIN_XC_FUNCTIONALS if spin_polarized else DEVICE_XC_FUNCTIONALS
    if mode == "off" or functional not in available:
        return False
    return mode == "on" or functional in DEFAULT_DEVICE_XC_FUNCTIONALS


def device_grid_supported(grid) -> bool:
    """A cluster grid with its lookup box, or an orthorhombic periodic cell.

    The non-orthogonal grid (``GeneralPeriodicGrid``) and the wire and slab
    grids keep the host functionals.
    """

    if getattr(grid, "is_general_periodic", False) or getattr(
        grid, "is_confined_periodic", False
    ):
        return False
    if _is_periodic(grid) and not getattr(grid, "is_orthorhombic", True):
        # A grid of Andrew Grieder's routines with oblique or unequal steps, or his slab: the gradients carry the
        # metric of the cell (V_xc.pbe._oblique_mix), which these kernels do not.
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
    return ("lda_spin_pw92",) if spin_polarized else (_LDA_KERNELS[functional],)


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
        # correlates with their negatives.
        weights = weights / grid.spacing
        self.weights = cp.asarray(weights)
        self.negated_weights = cp.asarray(-weights)
        self.flips = None
        if self.periodic:
            if reduction is not None:
                raise ValueError("a periodic grid has no symmetry reduction here")
            # pbe.py _to_box: the rows of a periodic cell are its box, row-major.
            positions = np.arange(box_size, dtype=np.int32)
            value_of_box = positions
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
                if flips and reduction.group_order > 1:
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
            np.int32(0 if self.flips is None else 1),
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
    (full-grid channels; each sees half of the frozen core density).
    """

    spin_polarized = True

    def __init__(self, functional: str, grid, **keywords) -> None:
        if keywords.get("reducer") is not None:
            raise ValueError("the spin channels are evaluated on the full grid")
        super().__init__(functional, grid, **keywords)

    def __call__(self, valence_up, valence_down) -> SpinXCResult:
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
            if functional == "pw92":
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
                potential_up=cp.asnumpy(potential_up),
                potential_down=cp.asnumpy(potential_down),
                energy_density=cp.asnumpy(energy_density),
                total_energy=total_energy,
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
    than CuPy, CA-LDA, a functional ``PARSEC_CUPY_XC`` leaves on the host, or
    a grid this module does not handle.  ``core_density`` returns the frozen
    core density of the present structure (default: that of ``reference``).
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
    if spin:
        # The loop of two channels evaluates the spin functional only.
        if not device_xc_selected(functional, spin_polarized=True):
            return None, None
        evaluator = DeviceSpinXCEvaluator(functional, grid, **keywords)
        return None, (ZeroAverageXC(evaluator) if zero_average else evaluator)
    if not device_xc_selected(functional):
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
]
