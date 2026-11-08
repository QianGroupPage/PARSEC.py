"""The semilocal part of a hybrid functional on a CUDA device (CuPy, float64).

The device counterpart of :func:`parsec_python.V_xc.hybrid.hybrid_semilocal_xc`
and :func:`~parsec_python.V_xc.hybrid.hybrid_semilocal_xc_spin`: the weighted
sum of semilocal components of a
:class:`~parsec_python.V_xc.semilocal_components.ComponentSemilocal`
(``hybrid_xc_kernels.cu``, line-by-line translations of the NumPy
components, compiled without fused multiply-adds) with the stencils, the
symmetry-orbit evaluation and the discrete variational potential of
:mod:`.device_xc`.  The short-range omega-PBE exchange of HSE and LC-wPBE
carries its derivatives by forward-mode differentiation on the device as on
the host (:mod:`parsec_python.V_xc.wpbe`).  The spin kernel works in libxc's
variables ``s_ab = D n_a . D n_b`` (:func:`parsec_python.V_xc.pbe.discrete_spin_gga_ab`).

Results agree with the host to round-off: the elementary functions are
CUDA's (``cbrt``, ``pow``, ``exp``, ``log``, ``log1p``, ``expm1``, ``erf``,
``erfcx``, ``asinh``, ``atan``) and the grid sums run in another order.
A semilocal evaluator that is not a ``ComponentSemilocal`` has no device
form here (:func:`device_hybrid_supported` tells); the semilocal part of a
libxc hybrid runs on the device in :mod:`.device_libxc`.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from parsec_python.V_xc import XCResult
from parsec_python.V_xc.ca_lda_spin import SpinXCResult
from parsec_python.V_xc.semilocal_components import ComponentSemilocal

from . import device_xc
from .device_xc import _DeviceEvaluatorBase, _StencilGeometry, _launch, device_grid_supported

#: Component order of the weights and omegas the kernels take.
COMPONENTS = (
    "slater_x",
    "pbe_x",
    "b88_x",
    "pw91_x",
    "wpbe_sr_x",
    "b88_sr_x",
    "pbe_c",
    "pw91_c",
    "pw92_c",
    "vwn_rpa_c",
    "vwn5_c",
    "lyp_c",
)
_KERNEL_NAMES = ("xc_gradient", "xc_adjoint", "hybrid_gga", "hybrid_spin_gga", "hybrid_points")


def _hybrid_constants() -> dict[str, object]:
    from importlib import import_module

    b3lyp = import_module("parsec_python.V_xc.b3lyp")
    becke_lyp = import_module("parsec_python.V_xc.becke_lyp")
    ityh = import_module("parsec_python.V_xc.ityh")
    wpbe = import_module("parsec_python.V_xc.wpbe")
    vwn_sets = [
        b3lyp._VWN_PARAMETERS["rpa"],
        b3lyp._VWN_PARAMETERS["vwn5"],
        b3lyp._VWN_FERROMAGNETIC["rpa"],
        b3lyp._VWN_FERROMAGNETIC["vwn5"],
        b3lyp._VWN_SPIN_STIFFNESS,
    ]
    constants: dict[str, object] = {
        "B88_BETA": b3lyp._B88_BETA,
        "B88_CS": becke_lyp._B88_CS,
        "SLATER_CX": b3lyp._SLATER_CX,
        "LYP_A": b3lyp._LYP_A,
        "LYP_B": b3lyp._LYP_B,
        "LYP_C": b3lyp._LYP_C,
        "LYP_D": b3lyp._LYP_D,
        "LYP_CF": b3lyp._LYP_CF,
        "LYP_CF_SPIN": becke_lyp._LYP_CF_SPIN,
        "VWN_F_SECOND": b3lyp._VWN_F_SECOND,
        "FZ_DENOMINATOR": becke_lyp._FZ_DENOMINATOR,
        "ITYH_X": ityh._X,
        "ITYH_SERIES_CUTOFF": ityh._SERIES_CUTOFF,
        "WP_A": wpbe._A,
        "WP_B": wpbe._B,
        "WP_C": wpbe._C,
        "WP_D": wpbe._D,
        "WP_E": wpbe._E,
        "WP_HA1": wpbe._HA1,
        "WP_HA2": wpbe._HA2,
        "WP_HA3": wpbe._HA3,
        "WP_HA4": wpbe._HA4,
        "WP_HA5": wpbe._HA5,
        "WP_FC1": wpbe._FC1,
        "WP_FC2": wpbe._FC2,
        "WP_EGA1": wpbe._EGA1,
        "WP_EGA2": wpbe._EGA2,
        "WP_EGA3": wpbe._EGA3,
        "WP_EG_CUTOFF": wpbe._EG_CUTOFF,
        "WP_NU_CUTOFF": wpbe._NU_CUTOFF,
        "WP_EB1": wpbe._EB1,
        "WP_S_TRANSITION": wpbe._S_TRANSITION,
        "WP_S_MAX": wpbe._S_MAX,
        "WP_S_CONST": wpbe._S_CONST,
        "WP_S_FLOOR": wpbe._S_FLOOR,
        "WP_S_DERIVATIVE_FLOOR": wpbe._S_DERIVATIVE_FLOOR,
        "WP_SQRT_PI": wpbe._SQRT_PI,
        "WP_SQRT_A": np.sqrt(wpbe._A),
        "WP_A15": wpbe._A**1.5,
        "WP_A25": wpbe._A**2.5,
        "WP_A2": wpbe._A**2,
        "WP_A3": wpbe._A**3,
        "WP_EULER": wpbe._EULER,
    }
    for index, value in enumerate(wpbe._EA, start=1):
        constants[f"WP_EA{index}"] = value
    constants["WP_E1_TERMS"] = int(wpbe._E1_SERIES_TERMS)
    constants["WP_E1_DEPTH"] = int(wpbe._E1_FRACTION_DEPTH)
    constants["VWN_SETS"] = "{" + ", ".join("{" + ", ".join(repr(float(v)) for v in row) + "}" for row in vwn_sets) + "}"
    constants["ITYH_SERIES"] = "{" + ", ".join(repr(float(v)) for v in ityh._SERIES) + "}"
    return constants


def hybrid_kernel_source() -> str:
    """``device_xc_kernels.cu`` and ``hybrid_xc_kernels.cu`` with the constants written in."""

    source = Path(__file__).with_name("hybrid_xc_kernels.cu").read_text()
    for name, value in _hybrid_constants().items():
        if isinstance(value, str):
            text = value
        elif isinstance(value, int):
            text = str(value)
        else:
            text = f"({float(value)!r})"
        source = source.replace(f"@{name}@", text)
    missing = re.findall(r"@[A-Z][A-Z0-9_]*@", source)
    if missing:
        raise RuntimeError(f"constants of the hybrid XC kernels without a value: {missing}")
    return device_xc.kernel_source() + "\n" + source


_MODULES: dict[int, dict[str, object]] = {}


def _hybrid_kernels(cp, device_id: int) -> dict[str, object]:
    kernels = _MODULES.get(int(device_id))
    if kernels is None:
        options = ("-std=c++11", "--fmad=false") + tuple(f"-DKERNEL_{name}" for name in _KERNEL_NAMES)
        with cp.cuda.Device(device_id):
            module = cp.RawModule(code=hybrid_kernel_source(), options=options)
            kernels = {name: module.get_function(name) for name in _KERNEL_NAMES}
        _MODULES[int(device_id)] = kernels
    return kernels


def component_parameters(semilocal) -> tuple[np.ndarray, np.ndarray]:
    """The kernel weights and omegas (in :data:`COMPONENTS` order) of a ``ComponentSemilocal``."""

    if not isinstance(semilocal, ComponentSemilocal):
        raise ValueError("the device evaluates the semilocal part of a ComponentSemilocal only")
    weights = np.zeros(len(COMPONENTS))
    omegas = np.zeros(len(COMPONENTS))
    for component in semilocal.components:
        index = COMPONENTS.index(component.name)
        if weights[index] != 0.0 and component.omega is not None and omegas[index] != component.omega:
            raise ValueError(f"{component.name} appears with two omegas; the device takes one")
        weights[index] += component.weight
        if component.omega is not None:
            omegas[index] = component.omega
    return weights, omegas


def device_hybrid_supported(functional, grid) -> bool:
    """Whether the semilocal part of ``functional`` has a device form on ``grid``."""

    return isinstance(getattr(functional, "semilocal", None), ComponentSemilocal) and device_grid_supported(grid)


class _HybridEvaluatorBase(_DeviceEvaluatorBase):
    """Device, kernels, geometry and the frozen core density of a hybrid's semilocal part."""

    def __init__(self, functional, grid, *, core_density=None, reducer=None, device_id=None) -> None:
        from ..backends.cupy import require_cupy

        if not device_grid_supported(grid):
            raise ValueError("the device XC evaluators need a cluster or orthorhombic grid")
        weights, omegas = component_parameters(functional.semilocal)
        cp, _ = require_cupy()
        self._cp = cp
        self.hybrid = functional
        self.functional = functional.label
        self.grid = grid
        self.reducer = reducer
        self.volume_element = float(grid.volume_element)
        self.device_id = int(cp.cuda.Device().id if device_id is None else device_id)
        self.cap = None
        self._core_source = core_density if callable(core_density) else (lambda: core_density)
        self._core_host = None
        self._core_device = None
        reduction = None if reducer is None else reducer.reduction
        with cp.cuda.Device(self.device_id):
            self._kernels = _hybrid_kernels(cp, self.device_id)
            self.geometry = _StencilGeometry(cp, grid, reduction, flips=True)
            self._multiplicities = (
                None
                if reducer is None
                else cp.asarray(np.asarray(reduction.multiplicities, dtype=np.float64))
            )
            self._weights = cp.asarray(weights)
            self._omegas = cp.asarray(omegas)
        self.size = int(grid.size if reducer is None else reducer.reduction.wedge_size)
        self.implementation_label = (
            f"CuPy float64 {'spin-polarized ' if self.spin_polarized else ''}semilocal part of {functional.name}"
            + (" on the symmetry orbits" if reducer is not None else "")
        )

    @property
    def planned_device_bytes(self) -> int:
        """What the evaluator keeps and the fields one evaluation holds (as a GGA of :mod:`.device_xc`)."""

        geometry = self.geometry.device_bytes
        weights = 0 if self._multiplicities is None else int(self._multiplicities.nbytes)
        core = 8 * self.size if self._core_source() is not None else 0
        fields = (19 if self.spin_polarized else 11) + (1 if self.spin_polarized and core else 0)
        return geometry + weights + core + 8 * self.size * fields


class DeviceHybridXCEvaluator(_HybridEvaluatorBase):
    """``evaluator(valence_density) -> XCResult``: the unpolarized semilocal part on a device (Rydberg).

    The counterpart of :func:`parsec_python.V_xc.hybrid.hybrid_semilocal_xc`;
    with ``reducer`` the density is a ``SymmetryScalarField`` (or a full-grid
    array) and the fields of the result are wedge fields, as in
    :class:`.device_xc.DeviceXCEvaluator`.
    """

    def __call__(self, valence_density) -> XCResult:
        cp = self._cp
        with cp.cuda.Device(self.device_id):
            density = cp.asarray(self._values(valence_density))
            core = self._core()
            if core is not None:
                density = density + core
            self._check(density, f"{self.hybrid.name} requires a nonnegative density")
            count = self.size
            kernels = self._kernels
            gradient = self.geometry.gradient(cp, kernels, density)
            energy_density = cp.empty(count, dtype=np.float64)
            epsilon = cp.empty(count, dtype=np.float64)
            f_n = cp.empty(count, dtype=np.float64)
            flux = cp.empty((3, count), dtype=np.float64)
            _launch(
                kernels["hybrid_gga"],
                count,
                density,
                gradient,
                self._weights,
                self._omegas,
                energy_density,
                f_n,
                flux,
                epsilon,
            )
            del gradient
            potential = self.geometry.adjoint(cp, kernels, flux, f_n, 2.0)
            total_energy = self._integral(energy_density)
            return XCResult(
                potential=self._field(potential),
                energy_per_electron=self._field(epsilon),
                energy_density=self._field(energy_density),
                total_energy=total_energy,
            )


class DeviceSpinHybridXCEvaluator(_HybridEvaluatorBase):
    """``evaluator(valence_up, valence_down) -> SpinXCResult`` on a device (Rydberg).

    The counterpart of :func:`parsec_python.V_xc.hybrid.hybrid_semilocal_xc_spin`
    (each channel sees half of the frozen core density): full-grid channels,
    or with ``reducer`` the orbit fields of a spin loop in the symmetry
    sectors, as :class:`.device_xc.DeviceSpinXCEvaluator` takes them.
    """

    spin_polarized = True

    def __call__(self, valence_up, valence_down) -> SpinXCResult:
        cp = self._cp
        with cp.cuda.Device(self.device_id):
            up = cp.asarray(self._values(valence_up))
            down = cp.asarray(self._values(valence_down))
            core = self._core()
            if core is not None:
                half = 0.5 * core
                up = up + half
                down = down + half
            message = f"{self.hybrid.name} requires nonnegative spin densities"
            self._check(up, message)
            self._check(down, message)
            up = cp.maximum(up, 0.0)
            down = cp.maximum(down, 0.0)
            count = self.size
            kernels = self._kernels
            gradient_up = self.geometry.gradient(cp, kernels, up)
            gradient_down = self.geometry.gradient(cp, kernels, down)
            energy_density = cp.empty(count, dtype=np.float64)
            f_up = cp.empty(count, dtype=np.float64)
            f_down = cp.empty(count, dtype=np.float64)
            flux_up = cp.empty((3, count), dtype=np.float64)
            flux_down = cp.empty((3, count), dtype=np.float64)
            _launch(
                kernels["hybrid_spin_gga"],
                count,
                up,
                down,
                gradient_up,
                gradient_down,
                self._weights,
                self._omegas,
                energy_density,
                f_up,
                f_down,
                flux_up,
                flux_down,
            )
            del gradient_up, gradient_down
            potential_up = self.geometry.adjoint(cp, kernels, flux_up, f_up, 2.0)
            potential_down = self.geometry.adjoint(cp, kernels, flux_down, f_down, 2.0)
            total_energy = self._integral(energy_density)
            return SpinXCResult(
                potential_up=self._field(potential_up),
                potential_down=self._field(potential_down),
                energy_density=self._field(energy_density),
                total_energy=total_energy,
            )


def device_semilocal_points(semilocal, *arrays, device_id: int | None = None) -> tuple[np.ndarray, ...]:
    """The pointwise partials on the device, for tests: ``(rho, sigma)`` or the five spin variables.

    Returns what ``semilocal.unpolarized`` or ``semilocal.polarized`` returns
    on the host.
    """

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    weights, omegas = component_parameters(semilocal)
    polarized = len(arrays) == 5
    if not polarized and len(arrays) != 2:
        raise ValueError("pass (rho, sigma) or (rho_up, rho_down, sigma_uu, sigma_ud, sigma_dd)")
    device = int(cp.cuda.Device().id if device_id is None else device_id)
    with cp.cuda.Device(device):
        kernels = _hybrid_kernels(cp, device)
        inputs = cp.asarray(np.ascontiguousarray(np.vstack([np.asarray(a, dtype=np.float64) for a in arrays])))
        count = int(inputs.shape[1])
        outputs = cp.empty((6 if polarized else 3, count), dtype=np.float64)
        _launch(
            kernels["hybrid_points"],
            count,
            np.int32(1 if polarized else 0),
            inputs,
            cp.asarray(weights),
            cp.asarray(omegas),
            outputs,
        )
        host = cp.asnumpy(outputs)
    return tuple(host)


__all__ = [
    "COMPONENTS",
    "DeviceHybridXCEvaluator",
    "DeviceSpinHybridXCEvaluator",
    "component_parameters",
    "device_hybrid_supported",
    "device_semilocal_points",
    "hybrid_kernel_source",
]
