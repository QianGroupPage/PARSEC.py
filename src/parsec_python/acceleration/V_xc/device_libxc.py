"""libxc's functionals on a CUDA device: the grid evaluators of the SCF (CuPy, float64).

The device counterparts of :func:`parsec_python.V_xc.semilocal.semilocal_xc`
and :func:`~parsec_python.V_xc.semilocal.semilocal_xc_spin` for a
:class:`parsec_python.V_xc.libxc.LibxcSemilocal` of family LDA or GGA (a
libxc functional, or the semilocal part of a libxc hybrid): the density
gradient ``D n`` and the transpose ``D^T`` of :mod:`.device_xc` (the
centered stencil of ``Expansion_Order``, zero outside a cluster, wrapped in
an orthorhombic cell, one value per symmetry orbit in the sectors), the
pointwise functional by libxc's CUDA build on the device pointers of the
CuPy arrays (libxc >= 7.1, ``XC_FLAGS_ON_DEVICE``), and the discrete
variational potential

``v = 2 [vrho + D^T (2 vsigma D n)]`` (Rydberg),
``v_a = 2 [vrho_a + D^T (2 vsigma_aa D n_a + vsigma_ab D n_b)]`` (spin),

with the frozen core density added (half per channel), as on the host.
The packing kernels (``device_libxc_kernels.cu``) are compiled without fused
multiply-adds and sum in the order of the NumPy code, so host and device
differ by libxc's elementary functions on the two processors (``pow``,
``exp``, ``log``, ``cbrt``, ...) and the order of the grid sums: round-off.

A meta-GGA is evaluated pointwise by the same ``LibxcSemilocal`` on CuPy
arrays (``supports_device``); its grid terms (tau of the orbitals, the
operator of ``vtau``) belong to the meta-GGA Hamiltonian.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from parsec_python.V_xc import XCResult
from parsec_python.V_xc.ca_lda_spin import SpinXCResult
from parsec_python.V_xc.libxc import LibxcSemilocal, is_libxc_functional, libxc_semilocal

from . import device_xc
from .device_xc import _DeviceEvaluatorBase, _StencilGeometry, _launch, device_grid_supported

_KERNEL_NAMES = ("libxc_sigma", "libxc_spin_inputs", "libxc_outputs", "libxc_spin_outputs")
_MODULES: dict[int, dict[str, object]] = {}
# pbe.py: the energy per electron is set where the density exceeds this.
_EPSILON_THRESHOLD = 1.0e-20


def _libxc_kernels(cp, device_id: int) -> dict[str, object]:
    kernels = _MODULES.get(int(device_id))
    if kernels is None:
        options = ("-std=c++11", "--fmad=false") + tuple(f"-DKERNEL_{name}" for name in _KERNEL_NAMES)
        source = Path(__file__).with_name("device_libxc_kernels.cu").read_text()
        with cp.cuda.Device(device_id):
            module = cp.RawModule(code=source, options=options)
            kernels = {name: module.get_function(name) for name in _KERNEL_NAMES}
        kernels.update(device_xc._kernels(cp, device_id, ("xc_gradient", "xc_adjoint")))
        _MODULES[int(device_id)] = kernels
    return kernels


def device_libxc_available() -> bool:
    """Whether the loaded libxc evaluates on the device (a CUDA build >= 7.1); False without libxc."""

    from parsec_python.V_xc.libxc_library import LibxcUnavailable, load_libxc

    try:
        return bool(load_libxc().device_capable)
    except LibxcUnavailable:
        return False


def device_libxc_supported(semilocal, grid) -> bool:
    """Whether the grid evaluators of this module take ``semilocal`` on ``grid``."""

    return (
        isinstance(semilocal, LibxcSemilocal)
        and semilocal.family in ("lda", "gga")
        and device_grid_supported(grid)
        and device_libxc_available()
    )


class _LibxcEvaluatorBase(_DeviceEvaluatorBase):
    """Device, kernels, stencil geometry, core density of a libxc functional (LDA or GGA)."""

    def __init__(self, semilocal, grid, *, name=None, core_density=None, reducer=None, device_id=None) -> None:
        from ..backends.cupy import require_cupy

        if isinstance(semilocal, str):
            semilocal = libxc_semilocal(semilocal)
        if not isinstance(semilocal, LibxcSemilocal) or semilocal.family not in ("lda", "gga"):
            raise ValueError("the device libxc grid evaluators take a libxc LDA or GGA")
        if not device_grid_supported(grid):
            raise ValueError("the device XC evaluators need a cluster or orthorhombic grid")
        if not device_libxc_available():
            raise ValueError("the loaded libxc has no device support (a CUDA build of libxc >= 7.1)")
        cp, _ = require_cupy()
        self._cp = cp
        self.semilocal = semilocal
        self.functional = semilocal.spec
        self.name = name or semilocal.spec
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
            self._kernels = _libxc_kernels(cp, self.device_id)
            self.geometry = _StencilGeometry(cp, grid, reduction, flips=True)
            self._multiplicities = (
                None
                if reducer is None
                else cp.asarray(np.asarray(reduction.multiplicities, dtype=np.float64))
            )
        self.size = int(grid.size if reducer is None else reducer.reduction.wedge_size)
        self.implementation_label = (
            f"libxc CUDA float64 {'spin-polarized ' if self.spin_polarized else ''}{self.name}"
            + (" on the symmetry orbits" if reducer is not None else "")
        )

    @property
    def planned_device_bytes(self) -> int:
        """What the evaluator keeps and the float64 fields one evaluation holds at once.

        Unpolarized: density, gradient (3), sigma, energy, vrho, vsigma and
        libxc's three outputs of a component, energy density, energy per
        electron, flux (3) and potential (18).  Spin: both channels with
        their gradients (8), libxc's interleaved inputs (5), sums (6) and the
        outputs of a component (6), energy density, the two ``df/dn`` and
        fluxes (8), the potentials (2): 36, one more with a core density.
        """

        geometry = self.geometry.device_bytes
        weights = 0 if self._multiplicities is None else int(self._multiplicities.nbytes)
        core = 8 * self.size if self._core_source() is not None else 0
        fields = (36 + (1 if core else 0)) if self.spin_polarized else 18
        return geometry + weights + core + 8 * self.size * fields


class DeviceLibxcXCEvaluator(_LibxcEvaluatorBase):
    """``evaluator(valence_density) -> XCResult``: an unpolarized libxc LDA/GGA on a device (Rydberg).

    The counterpart of :func:`parsec_python.V_xc.semilocal.semilocal_xc`;
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
            self._check(density, f"{self.name} requires a nonnegative density")
            count = self.size
            kernels = self._kernels
            gradient = self.geometry.gradient(cp, kernels, density)
            sigma = cp.empty(count, dtype=np.float64)
            _launch(kernels["libxc_sigma"], count, gradient, sigma)
            energy, vrho, vsigma, _, _ = self.semilocal.evaluate_layout(density, sigma, polarized=False)
            del sigma
            energy_density = cp.empty(count, dtype=np.float64)
            epsilon = cp.empty(count, dtype=np.float64)
            flux = cp.empty((3, count), dtype=np.float64)
            _launch(
                kernels["libxc_outputs"],
                count,
                density,
                gradient,
                energy,
                vsigma,
                np.float64(_EPSILON_THRESHOLD),
                energy_density,
                flux,
                epsilon,
            )
            del gradient, energy, vsigma
            potential = self.geometry.adjoint(cp, kernels, flux, vrho, 2.0)
            total_energy = self._integral(energy_density)
            return XCResult(
                potential=self._field(potential),
                energy_per_electron=self._field(epsilon),
                energy_density=self._field(energy_density),
                total_energy=total_energy,
            )


class DeviceSpinLibxcXCEvaluator(_LibxcEvaluatorBase):
    """``evaluator(valence_up, valence_down) -> SpinXCResult``: a spin libxc LDA/GGA on a device (Rydberg).

    The counterpart of :func:`parsec_python.V_xc.semilocal.semilocal_xc_spin`
    (each channel sees half of the frozen core density): full-grid channels,
    or with ``reducer`` the orbit fields of a spin loop in the symmetry
    sectors.
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
            message = f"{self.name} requires nonnegative spin densities"
            self._check(up, message)
            self._check(down, message)
            up = cp.maximum(up, 0.0)
            down = cp.maximum(down, 0.0)
            count = self.size
            kernels = self._kernels
            gradient_up = self.geometry.gradient(cp, kernels, up)
            gradient_down = self.geometry.gradient(cp, kernels, down)
            rho = cp.empty((count, 2), dtype=np.float64)
            sigma = cp.empty((count, 3), dtype=np.float64)
            _launch(kernels["libxc_spin_inputs"], count, up, down, gradient_up, gradient_down, rho, sigma)
            del up, down
            energy, vrho, vsigma, _, _ = self.semilocal.evaluate_layout(rho, sigma, polarized=True)
            del rho, sigma
            energy_density = cp.empty(count, dtype=np.float64)
            f_up = cp.empty(count, dtype=np.float64)
            f_down = cp.empty(count, dtype=np.float64)
            flux_up = cp.empty((3, count), dtype=np.float64)
            flux_down = cp.empty((3, count), dtype=np.float64)
            _launch(
                kernels["libxc_spin_outputs"],
                count,
                gradient_up,
                gradient_down,
                energy,
                vrho,
                vsigma,
                energy_density,
                f_up,
                f_down,
                flux_up,
                flux_down,
            )
            del gradient_up, gradient_down, energy, vrho, vsigma
            potential_up = self.geometry.adjoint(cp, kernels, flux_up, f_up, 2.0)
            potential_down = self.geometry.adjoint(cp, kernels, flux_down, f_down, 2.0)
            total_energy = self._integral(energy_density)
            return SpinXCResult(
                potential_up=self._field(potential_up),
                potential_down=self._field(potential_down),
                energy_density=self._field(energy_density),
                total_energy=total_energy,
            )


def libxc_xc_evaluators(reference, *, spec: str, reducer=None, device_id=None, core_density=None):
    """``(unpolarized, spin)`` device evaluators of the libxc functional ``spec`` (one of them ``None``).

    ``(None, None)`` where the device does not take it: a meta-GGA (its grid
    terms are the meta-GGA Hamiltonian's), a grid without device stencils, or
    a libxc without device support (the host evaluates libxc then).
    """

    scf = reference.input.scf
    if not is_libxc_functional(spec):
        raise ValueError(f"{spec!r} is not a libxc functional")
    semilocal = libxc_semilocal(spec)
    if not device_libxc_supported(semilocal, reference.grid):
        return None, None
    keywords = {
        "core_density": core_density if core_density is not None else (lambda: reference.core_density),
        "reducer": reducer,
        "device_id": device_id,
    }
    if bool(getattr(scf, "spin_polarized", False)):
        return None, DeviceSpinLibxcXCEvaluator(semilocal, reference.grid, **keywords)
    return DeviceLibxcXCEvaluator(semilocal, reference.grid, **keywords), None


__all__ = [
    "DeviceLibxcXCEvaluator",
    "DeviceSpinLibxcXCEvaluator",
    "device_libxc_available",
    "device_libxc_supported",
    "libxc_xc_evaluators",
]
