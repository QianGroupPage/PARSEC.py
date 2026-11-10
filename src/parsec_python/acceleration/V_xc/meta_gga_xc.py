"""The meta-GGA functional of an accelerated SCF, for fields on the full grid or on symmetry orbits.

The reference evaluation (:func:`parsec_python.V_xc.meta_gga.meta_gga_xc`,
``meta_gga_xc_spin``) on full-grid host arrays; fields that the SCF keeps on
one value per symmetry orbit (``SymmetrySCFReducer``) are expanded for it and
the potentials and ``vtau`` reduced again, an exact representation change for
invariant fields (as the host functionals of the sectors are evaluated,
``acceleration/driver.py``).  ``reference_of()`` returns the reference system
of the present atoms.

The orbital form of the density gradient (``density_gradient=`` of the
calls): ``g`` and ``W = de/dg`` are the Cartesian components at the orbit
representatives (:class:`~parsec_python.acceleration.Hamiltonian.meta_gga_device.OrbitVectorFields`),
which the host path expands with the operations of the group; the device
path evaluates the functional pointwise on the orbit values, adds the
gradient of the core density at the representatives and forms ``dE/dn_core``
with the transposed stencil of the GGA flux.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np


def _vector_fields(reference, reducer):
    from ..Hamiltonian.meta_gga_device import orbit_vector_fields

    return orbit_vector_fields(reference.grid, reducer.reduction)


def _expanded(reference, reducer, components):
    values = np.stack([np.asarray(reducer.wedge_values(component), dtype=np.float64) for component in components])
    return tuple(_vector_fields(reference, reducer).expand(values))


def _reduced(reference, reducer, components):
    reduced = _vector_fields(reference, reducer).reduce(np.stack([np.asarray(row) for row in components]))
    return tuple(reducer.field(np.ascontiguousarray(row)) for row in reduced)


class HostMetaGGAOnOrbits:
    """``evaluate_xc(density, tau=...)`` of a meta-GGA for an accelerated system (unpolarized)."""

    def __init__(self, reference_of, reducer=None) -> None:
        self.reference_of = reference_of
        self.reducer = reducer

    def __call__(self, density, tau=None, density_gradient=None):
        reference = self.reference_of()
        if tau is None:
            raise ValueError("a meta-GGA needs the kinetic-energy density of the orbitals")
        reducer = self.reducer
        if reducer is None:
            return reference.evaluate_xc(density, tau=tau, density_gradient=density_gradient)
        gradient = None if density_gradient is None else _expanded(reference, reducer, density_gradient)
        result = reference.evaluate_xc(reducer.to_full(density), tau=reducer.to_full(tau), density_gradient=gradient)
        extra = {}
        if gradient is not None:
            extra = dict(
                gradient_flux=_reduced(reference, reducer, result.gradient_flux),
                core_potential=None if result.core_potential is None else reducer.from_full(result.core_potential),
            )
        return replace(
            result,
            potential=reducer.from_full(result.potential),
            energy_per_electron=reducer.from_full(result.energy_per_electron),
            energy_density=reducer.from_full(result.energy_density),
            kinetic_potential=reducer.from_full(result.kinetic_potential),
            **extra,
        )


class HostMetaGGASpinOnOrbits:
    """``evaluate_spin_xc(up, down, tau_up=..., tau_down=...)`` of a meta-GGA for an accelerated system."""

    def __init__(self, reference_of, reducer=None) -> None:
        self.reference_of = reference_of
        self.reducer = reducer

    def __call__(self, density_up, density_down, tau_up=None, tau_down=None, density_gradients=None):
        reference = self.reference_of()
        if tau_up is None or tau_down is None:
            raise ValueError("a meta-GGA needs the kinetic-energy densities of both channels")
        reducer = self.reducer
        if reducer is None:
            return reference.evaluate_spin_xc(
                density_up, density_down, tau_up=tau_up, tau_down=tau_down, density_gradients=density_gradients
            )
        full = reducer.to_full
        gradients = None
        if density_gradients is not None:
            gradients = tuple(_expanded(reference, reducer, channel) for channel in density_gradients)
        result = reference.evaluate_spin_xc(
            full(density_up), full(density_down), tau_up=full(tau_up), tau_down=full(tau_down),
            density_gradients=gradients,
        )
        extra = {}
        if gradients is not None:
            extra = dict(
                gradient_flux_up=_reduced(reference, reducer, result.gradient_flux_up),
                gradient_flux_down=_reduced(reference, reducer, result.gradient_flux_down),
                core_potential=None if result.core_potential is None else reducer.from_full(result.core_potential),
            )
        return replace(
            result,
            potential_up=reducer.from_full(result.potential_up),
            potential_down=reducer.from_full(result.potential_down),
            kinetic_potential_up=reducer.from_full(result.kinetic_potential_up),
            kinetic_potential_down=reducer.from_full(result.kinetic_potential_down),
            **extra,
        )


class DeviceMetaGGAOnOrbits:
    """A meta-GGA on a CUDA device, on full-grid fields or one value per symmetry orbit.

    The evaluator (``supports_device``, e.g. :class:`parsec_python.V_xc.scan.NativeScan`)
    runs on CuPy arrays; the density gradient and the transpose of the GGA
    flux are those of the device functionals (:mod:`.device_xc`: the stencil of
    ``density_gradient`` on orbit values, the flux of an orbit carried by the
    reflection signs), so the potential is that of
    :func:`parsec_python.V_xc.meta_gga.meta_gga_xc` to round-off.  ``vtau``
    is pointwise.  Evaluators with ``needs_laplacian`` take the host path.
    ``spin`` makes the call ``(up, down, tau_up=, tau_down=)``.  In the
    orbital form (``density_gradient=`` / ``density_gradients=``) the
    potential is ``vrho`` alone, the result holds ``W`` and ``dE/dn_core``,
    and the gradient of the core density at the values is added to ``g``.
    The orbit values of a group with operations that exchange axes (signed
    permutations) take the host path: the stencil's reflection signs do not
    carry their fluxes.
    """

    def __init__(self, evaluator, grid, *, reducer=None, device_id: int = 0, core_density=None, spin: bool = False) -> None:
        import numpy as np

        from ..backends.cupy import require_cupy
        from .device_xc import _StencilGeometry, _kernels, device_grid_supported

        if getattr(evaluator, "needs_laplacian", False) or not getattr(evaluator, "supports_device", False):
            raise ValueError("this meta-GGA has no device evaluation")
        if not device_grid_supported(grid):
            raise ValueError("the device XC evaluators need a cluster or orthorhombic grid")
        cp, _ = require_cupy()
        self._cp = cp
        self.evaluator = evaluator
        self.grid = grid
        self.reducer = reducer
        self.spin = bool(spin)
        self.device_id = int(device_id)
        self.volume_element = float(grid.volume_element)
        self._core_source = core_density if callable(core_density) else (lambda: core_density)
        self._core_host = None
        self._core_device = None
        reduction = None if reducer is None else reducer.reduction
        operations = getattr(reduction, "operations", None)
        if operations is not None and not all(
            np.array_equal(operation, np.diag(np.diag(operation))) for operation in np.asarray(operations)
        ):
            raise ValueError("the device meta-GGA evaluation takes groups of reflections")
        with cp.cuda.Device(self.device_id):
            self._kernels = _kernels(cp, self.device_id, ("xc_adjoint", "xc_gradient"))
            self.geometry = _StencilGeometry(cp, grid, reduction, flips=True)
            self._multiplicities = (
                None if reducer is None else cp.asarray(np.asarray(reduction.multiplicities, dtype=np.float64))
            )
        self.size = int(grid.size if reducer is None else reduction.wedge_size)
        self.statistics = dict(calls=0, seconds=0.0)

    @property
    def planned_device_bytes(self) -> int:
        """The geometry and about 50 fields of the values during one evaluation (complex-step copies; g, W and dE/dn_core of the orbital form)."""

        return int(self.geometry.device_bytes + 8 * self.size * 50)

    def _values(self, field):
        import numpy as np

        values = self.reducer.wedge_values(field) if self.reducer is not None else np.asarray(field, dtype=np.float64)
        values = np.ascontiguousarray(values, dtype=np.float64)
        if values.shape != (self.size,):
            raise ValueError("the field does not match the XC grid")
        return self._cp.asarray(values)

    def _core(self):
        core = self._core_source()
        if core is None:
            self._core_host = self._core_device = None
            return None
        if core is not self._core_host:
            self._core_device = self._values(core)
            self._core_host = core
        return self._core_device

    def _integral(self, values) -> float:
        cp = self._cp
        total = cp.sum(values) if self._multiplicities is None else cp.dot(self._multiplicities, values)
        return float(self.volume_element * float(total))

    def _field(self, values):
        host = self._cp.asnumpy(values)
        return host if self.reducer is None else self.reducer.field(host)

    def _clamped_count(self, sigma, density, tau) -> int:
        cp = self._cp
        mask = (density > 1.0e-20) & (sigma > 8.0 * density * tau)
        if self._multiplicities is None:
            return int(cp.count_nonzero(mask))
        return int(cp.sum(self._multiplicities * mask))

    def _given_gradient(self, components, core, scale: float):
        """``g`` of the orbital form at the values ``(3, count)``, with ``scale`` times the core density's gradient."""

        cp = self._cp
        gradient = cp.stack([self._values(component) for component in components])
        if core is not None:
            gradient = gradient + self.geometry.gradient(cp, self._kernels, scale * core)
        return gradient

    def _vector(self, values):
        return tuple(self._field(values[axis]) for axis in range(3))

    def __call__(self, *densities, tau=None, tau_up=None, tau_down=None, density_gradient=None, density_gradients=None):
        from time import perf_counter

        from parsec_python.V_xc.meta_gga import (
            MetaGGASpinXCResult,
            MetaGGAXCResult,
            _floored,
            tau_floor_options,
            weizsaecker_chain,
            weizsaecker_floor,
        )

        cp = self._cp
        started = perf_counter()
        geometry, kernels = self.geometry, self._kernels
        with cp.cuda.Device(self.device_id):
            core = self._core()
            if not self.spin:
                (valence,) = densities
                if tau is None:
                    raise ValueError("a meta-GGA needs the kinetic-energy density of the orbitals")
                density = self._values(valence)
                if core is not None:
                    density = density + core
                kinetic = self._values(tau)
                orbital = density_gradient is not None
                gradient = (
                    self._given_gradient(density_gradient, core, 1.0)
                    if orbital
                    else geometry.gradient(cp, kernels, density)
                )
                sigma = cp.sum(gradient * gradient, axis=0)
                floored = _floored(self.evaluator)
                given, below = (
                    weizsaecker_floor(cp, density, sigma, kinetic, **tau_floor_options(self.evaluator))
                    if floored
                    else (kinetic, None)
                )
                energy, vrho, vsigma, vtau, _ = self.evaluator.unpolarized(density, sigma, given, None)
                if floored:
                    vrho, vsigma, vtau = weizsaecker_chain(cp, density, sigma, below, vrho, vsigma, vtau)
                flux = cp.ascontiguousarray(2.0 * vsigma[None, :] * gradient)
                vrho = cp.ascontiguousarray(vrho)
                extra = {}
                if orbital:
                    potential = 2.0 * vrho
                    extra = dict(
                        gradient_flux=self._vector(flux),
                        core_potential=self._field(geometry.adjoint(cp, kernels, flux, vrho, 2.0)),
                    )
                else:
                    potential = geometry.adjoint(cp, kernels, flux, vrho, 2.0)
                energy_density = 2.0 * energy
                positive = density > 1.0e-20
                epsilon = cp.where(positive, energy_density / cp.where(positive, density, 1.0), 0.0)
                result = MetaGGAXCResult(
                    potential=self._field(potential),
                    energy_per_electron=self._field(epsilon),
                    energy_density=self._field(energy_density),
                    total_energy=self._integral(energy_density),
                    kinetic_potential=self._field(vtau),
                    fermi_hole_clamped=self._clamped_count(sigma, density, kinetic),
                    **extra,
                )
            else:
                valence_up, valence_down = densities
                if tau_up is None or tau_down is None:
                    raise ValueError("a meta-GGA needs the kinetic-energy densities of both channels")
                up = self._values(valence_up)
                down = self._values(valence_down)
                if core is not None:
                    up = up + 0.5 * core
                    down = down + 0.5 * core
                up = cp.maximum(up, 0.0)
                down = cp.maximum(down, 0.0)
                kinetic_up = self._values(tau_up)
                kinetic_down = self._values(tau_down)
                orbital = density_gradients is not None
                if orbital:
                    grad_up = self._given_gradient(density_gradients[0], core, 0.5)
                    grad_down = self._given_gradient(density_gradients[1], core, 0.5)
                else:
                    grad_up = geometry.gradient(cp, kernels, up)
                    grad_down = geometry.gradient(cp, kernels, down)
                s_uu = cp.sum(grad_up * grad_up, axis=0)
                s_ud = cp.sum(grad_up * grad_down, axis=0)
                s_dd = cp.sum(grad_down * grad_down, axis=0)
                floored = _floored(self.evaluator)
                options = tau_floor_options(self.evaluator)
                given_up, below_up = (
                    weizsaecker_floor(cp, up, s_uu, kinetic_up, **options) if floored else (kinetic_up, None)
                )
                given_down, below_down = (
                    weizsaecker_floor(cp, down, s_dd, kinetic_down, **options) if floored else (kinetic_down, None)
                )
                (energy, v_up, v_down, vs_uu, vs_ud, vs_dd, vt_up, vt_down, _, _) = self.evaluator.polarized(
                    up, down, s_uu, s_ud, s_dd, given_up, given_down, None, None
                )
                if floored:
                    v_up, vs_uu, vt_up = weizsaecker_chain(cp, up, s_uu, below_up, v_up, vs_uu, vt_up)
                    v_down, vs_dd, vt_down = weizsaecker_chain(cp, down, s_dd, below_down, v_down, vs_dd, vt_down)
                flux_up = cp.ascontiguousarray(2.0 * vs_uu[None, :] * grad_up + vs_ud[None, :] * grad_down)
                flux_down = cp.ascontiguousarray(2.0 * vs_dd[None, :] * grad_down + vs_ud[None, :] * grad_up)
                v_up = cp.ascontiguousarray(v_up)
                v_down = cp.ascontiguousarray(v_down)
                extra = {}
                if orbital:
                    potential_up = 2.0 * v_up
                    potential_down = 2.0 * v_down
                    # dE/dn_core (Ry): half of the core density in each channel.
                    core_potential = geometry.adjoint(cp, kernels, flux_up, v_up, 1.0) + geometry.adjoint(
                        cp, kernels, flux_down, v_down, 1.0
                    )
                    extra = dict(
                        gradient_flux_up=self._vector(flux_up),
                        gradient_flux_down=self._vector(flux_down),
                        core_potential=self._field(core_potential),
                    )
                else:
                    potential_up = geometry.adjoint(cp, kernels, flux_up, v_up, 2.0)
                    potential_down = geometry.adjoint(cp, kernels, flux_down, v_down, 2.0)
                energy_density = 2.0 * energy
                result = MetaGGASpinXCResult(
                    potential_up=self._field(potential_up),
                    potential_down=self._field(potential_down),
                    energy_density=self._field(energy_density),
                    total_energy=self._integral(energy_density),
                    kinetic_potential_up=self._field(vt_up),
                    kinetic_potential_down=self._field(vt_down),
                    fermi_hole_clamped=self._clamped_count(s_uu, up, kinetic_up)
                    + self._clamped_count(s_dd, down, kinetic_down),
                    **extra,
                )
        self.statistics["calls"] += 1
        self.statistics["seconds"] += perf_counter() - started
        return result


__all__ = ["DeviceMetaGGAOnOrbits", "HostMetaGGAOnOrbits", "HostMetaGGASpinOnOrbits"]
