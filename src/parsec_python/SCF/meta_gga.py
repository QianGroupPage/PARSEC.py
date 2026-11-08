"""The kinetic-energy-density part of an SCF loop of a meta-GGA.

:func:`parsec_python.SCF.run_scf` and :func:`parsec_python.SCF.spin_polarized.run_spin_scf`
run a meta-GGA (``SCFSettings.is_meta_gga``) with :class:`MetaGGALoop`, which
adds to PARSEC's potential-mixing loop:

* the first input potential holds PBE (spin PBE) of the start density and no
  orbital terms (``vtau_in = 0``, ``W_in = 0``), as PARSEC 1.5 starts its
  libxc meta-GGAs: there are no orbitals before the first eigensolve;
* every Hamiltonian holds the orbital terms of its channel
  (:class:`parsec_python.V_xc.meta_gga.KineticPotentialOperator`):
  ``-div(vtau_in grad)`` and, in the orbital form of the density gradient
  (:func:`parsec_python.V_xc.meta_gga.meta_gga_gradient_form`: by default for
  every functional smooth across ``tau = tau_W``), the first-order term
  ``2 sum_c [W_c G_c + G_c^T W_c]`` of ``W = de/dg``;
* the orbitals of an iteration give ``tau`` (and ``g``) beside the density,
  and the functional gives the local output potential, ``vtau_out`` (and
  ``W_out``);
* the mixer of ``Mixing_Method`` sees the joined vector ``(V, vtau, W)`` of
  all channels, one history for all (Anderson's least squares weighted by the
  density, :func:`meta_gga_anderson`); ``PARSEC_META_GGA_MIXING=output`` instead
  takes ``vtau_out`` unmixed into the next Hamiltonian (density form only);
* the stop test stays PARSEC's on ``V`` (the residual of ``vtau`` is recorded);
* the energy removes the band-energy parts of the orbital terms,
  ``2 h^3 sum vtau_in tau + 2 h^3 sum W_in . g``
  (:class:`parsec_python.models.MetaGGAEnergyBreakdown`).

The system supplies the pieces through optional members, with host defaults
on the full grid: ``kinetic_energy_density(wavefunctions, occupations,
degeneracy, gradient=False)`` (``tau``, with ``gradient`` also the three
components of ``g``, where the orbitals are), ``evaluate_xc(density, tau=tau,
density_gradient=g)``, ``evaluate_spin_xc(up, down, tau_up=..., tau_down=...,
density_gradients=(g_up, g_down))``, ``evaluate_meta_gga_start_xc(density)`` /
``evaluate_meta_gga_start_spin_xc(up, down)`` (PBE) and ``hamiltonian(V,
kinetic_potential=vtau, gradient_flux=W)``.  A loop that keeps its scalar
fields on symmetry orbits (``scalar_field_adapter``) holds the components of
``g`` and ``W`` as orbit fields too (their values at the representative points,
along the derivative directions of the sectors), joins the blocks with
``adapter.join_blocks`` / ``split_blocks`` and mixes with
``adapter.block_mixer(settings, blocks)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os

import numpy as np

from ..models import MetaGGAEnergyBreakdown

_MIXING_CHOICES = ("joint", "output", "linear")


def meta_gga_anderson() -> str:
    """``PARSEC_META_GGA_ANDERSON``: ``weighted`` (default) or ``plain``.

    ``weighted``: the least squares of Anderson mixing (``anderson.f90``)
    weigh every value of ``(V, vtau, W)`` by the density there (times the
    orbit multiplicity in the symmetry sectors), the metric of PARSEC's stop
    test (the weighted SRE).  A meta-GGA's potential reacts strongly to the
    orbitals where the density is low (``alpha`` of SCAN is a ratio of small
    numbers there); with plain inner products those values dominate the least
    squares and the step of the rest becomes small.  H2O, SCAN, density form,
    h = 0.16 bohr on an A100: SRE 2e-4 after 21 iterations (``plain``: 85;
    PBE: 18).  ``plain`` is PARSEC's metric, as for every other functional.
    """

    value = os.environ.get("PARSEC_META_GGA_ANDERSON", "weighted").strip().lower()
    if value not in ("weighted", "plain"):
        raise ValueError("PARSEC_META_GGA_ANDERSON must be weighted or plain")
    return value


def meta_gga_mixing() -> str:
    """``PARSEC_META_GGA_MIXING``: ``joint`` (default), ``output`` or ``linear`` (see the module)."""

    value = os.environ.get("PARSEC_META_GGA_MIXING", "joint").strip().lower()
    if value not in _MIXING_CHOICES:
        raise ValueError("PARSEC_META_GGA_MIXING must be joint, output or linear")
    return value


@dataclass(frozen=True)
class MetaGGARecord:
    """What a meta-GGA SCF loop leaves in its result (full-grid fields, one per channel).

    ``kinetic_potentials`` is the ``vtau`` of the last eigenproblem,
    ``output_kinetic_potentials`` that of the last orbitals,
    ``next_kinetic_potentials`` the mixed one a further iteration (or the
    next structure of an ionic loop) starts from, ``kinetic_energy_densities``
    the ``tau`` of the last orbitals (Hartree).  ``kinetic_residual`` is
    ``max_s sqrt(h^3 sum (vtau_out - vtau_in)^2)`` of the last iteration,
    ``fermi_hole_clamped`` the number of points with ``tau < tau_W`` there and
    ``mixing`` the choice of ``PARSEC_META_GGA_MIXING``.  In the orbital form
    ``gradient`` is ``"orbital"``, ``next_gradient_fluxes`` the mixed ``W`` per
    channel (three full-grid Cartesian components, the start of the next
    structure, on the full grid or in the sectors) and ``core_potential`` the
    ``dE/dn_core`` (Ry, full grid where the loop forms it) of the
    core-correction force; ``None`` otherwise.
    """

    kinetic_potentials: tuple
    output_kinetic_potentials: tuple
    next_kinetic_potentials: tuple
    kinetic_energy_densities: tuple
    kinetic_residual: float
    fermi_hole_clamped: int
    mixing: str
    gradient: str = "density"
    next_gradient_fluxes: tuple | None = None
    core_potential: object | None = None


def scf_settings_of(system):
    """The SCF settings of a prepared system (an accelerated one holds them in its reference)."""

    settings = getattr(getattr(system, "input", None), "scf", None)
    if settings is None:
        settings = getattr(getattr(getattr(system, "reference", None), "input", None), "scf", None)
    if settings is None:
        raise ValueError("the system holds no SCF settings")
    return settings


class MetaGGALoop:
    """The ``tau``, ``vtau`` (and ``g``, ``W``) of an SCF loop with ``channels`` spin channels (see the module)."""

    def __init__(self, system, *, channels: int, adapter=None, warm_start=None) -> None:
        from ..V_xc.meta_gga import meta_gga_gradient_form

        if channels not in (1, 2):
            raise ValueError("a meta-GGA loop has one or two spin channels")
        self.system = system
        self.channels = int(channels)
        self.adapter = adapter
        self.mixing = meta_gga_mixing()
        form, self.gradient_reason = meta_gga_gradient_form(scf_settings_of(system))
        self.orbital_gradient = form == "orbital"
        if self.orbital_gradient and self.mixing != "joint":
            raise ValueError("the orbital form of the density gradient mixes (V, vtau, W) jointly")
        self._to_internal = (lambda values: values) if adapter is None else adapter.from_full
        self._to_public = (lambda values: values) if adapter is None else adapter.to_full
        self.size = int(system.grid.size)
        given = None if warm_start is None else getattr(warm_start, "kinetic_potential", None)
        if given is None:
            # PARSEC 1.5: the first Hamiltonian has no orbital term.
            self.input = [self._zero() for _ in range(self.channels)]
            self.input_flux = [tuple(self._zero_values() for _ in range(3)) for _ in range(self.channels)]
            self.input_is_zero = True
        else:
            fields = (given,) if self.channels == 1 and not isinstance(given, tuple) else tuple(given)
            if len(fields) != self.channels:
                raise ValueError("the warm start holds the vtau of another number of spin channels")
            self.input = [self._to_internal(np.asarray(field, dtype=np.float64)) for field in fields]
            fluxes = getattr(warm_start, "gradient_flux", None)
            if self.orbital_gradient and fluxes is not None:
                if len(fluxes) != self.channels:
                    raise ValueError("the warm start holds the W of another number of spin channels")
                self.input_flux = [self._flux_to_internal(flux) for flux in fluxes]
            else:
                self.input_flux = [tuple(self._zero_values() for _ in range(3)) for _ in range(self.channels)]
            self.input_is_zero = False
        self.pending = None
        self.pending_flux = None
        self.outputs = None
        self.output_flux = None
        self.taus = None
        self.gradients = [None] * self.channels
        self.residual = float("nan")
        self.clamped = 0
        self.core_potential = None

    def _zero(self):
        return self._to_internal(np.zeros(self.size, dtype=np.float64))

    def _vector_fields(self):
        from ..acceleration.Hamiltonian.meta_gga_device import orbit_vector_fields

        return orbit_vector_fields(self.system.grid, self.adapter.reduction)

    def _flux_to_internal(self, flux):
        """The three components of ``W`` in the loop's layout, from full-grid Cartesian components (or already in it)."""

        components = tuple(flux)
        if len(components) != 3:
            raise ValueError("W has three Cartesian components")
        if self.adapter is None:
            return tuple(np.asarray(component, dtype=np.float64) for component in components)
        if all(isinstance(component, np.ndarray) and component.shape == (self.size,) for component in components):
            reduced = self._vector_fields().reduce(np.stack(components))
            return tuple(self.adapter.field(np.ascontiguousarray(row)) for row in reduced)
        return components

    def _flux_to_public(self, flux):
        """The full-grid Cartesian components of ``W`` in the loop's layout."""

        if self.adapter is None:
            return tuple(np.asarray(component, dtype=np.float64) for component in flux)
        values = np.stack([np.asarray(self.adapter.wedge_values(component), dtype=np.float64) for component in flux])
        return tuple(self._vector_fields().expand(values))

    def _zero_values(self):
        """A zero component of a vector field in the loop's layout (orbit values in the sectors)."""

        if self.adapter is None:
            return np.zeros(self.size, dtype=np.float64)
        return self.adapter.field(np.zeros(int(self.adapter.reduction.wedge_size), dtype=np.float64))

    @property
    def starts_with_pbe(self) -> bool:
        """Whether the first input potential is PBE's (no ``vtau`` taken over from a warm start)."""

        return self.input_is_zero

    # ----- the first potentials ------------------------------------------------

    def start_xc(self, density):
        """PBE of the start density (unpolarized), on the system's own path where it has one."""

        evaluate = getattr(self.system, "evaluate_meta_gga_start_xc", None)
        if callable(evaluate):
            return evaluate(density)
        from ..V_xc import pbe

        result = pbe(self._to_public(density), self.system.grid, self.system.core_density)
        if self.adapter is None:
            return result
        return replace(result, potential=self._to_internal(result.potential))

    def start_spin_xc(self, density_up, density_down):
        """Spin PBE of the start densities."""

        evaluate = getattr(self.system, "evaluate_meta_gga_start_spin_xc", None)
        if callable(evaluate):
            return evaluate(density_up, density_down)
        from ..V_xc import pbe_spin

        result = pbe_spin(
            self._to_public(density_up), self._to_public(density_down), self.system.grid, self.system.core_density
        )
        if self.adapter is None:
            return result
        return replace(
            result,
            potential_up=self._to_internal(result.potential_up),
            potential_down=self._to_internal(result.potential_down),
        )

    # ----- the Hamiltonian ----------------------------------------------------------

    def kinetic_potential(self, channel: int = 0):
        """``vtau_in`` of a channel for its Hamiltonian, ``None`` while it is zero."""

        return None if self.input_is_zero else self.input[channel]

    def flux_keywords(self, channel: int) -> dict:
        """``gradient_flux`` (``W_in``) of a channel's Hamiltonian in the orbital form, else nothing."""

        if not self.orbital_gradient or self.input_is_zero:
            return {}
        return {"gradient_flux": self.input_flux[channel]}

    # ----- tau and g ----------------------------------------------------------------

    def tau(self, wavefunctions, occupations, degeneracy: float, channel: int = 0):
        """``tau`` of a channel's orbitals, where the system keeps its fields (and ``g`` in the orbital form)."""

        build = getattr(self.system, "kinetic_energy_density", None)
        if not callable(build):
            raise NotImplementedError("this system cannot form the kinetic-energy density of its orbitals")
        if not self.orbital_gradient:
            return build(wavefunctions, occupations, degeneracy)
        tau, gradient = build(wavefunctions, occupations, degeneracy, gradient=True)
        self.gradients[channel] = tuple(gradient)
        return tau

    def symmetrize_gradient(self, channel: int, density_symmetry) -> None:
        """``Symmetrize_Charge_Density``: average ``g`` of a channel over the point group as ``n`` and ``tau`` are.

        ``density_symmetry`` is the group of the loop (a full-grid
        :class:`parsec_python.Grid.point_group.GridPointGroup`, or one on
        sector orbits with its ``group``); on orbits the Cartesian components
        at the representatives are expanded, averaged and taken back.
        """

        if not self.orbital_gradient or density_symmetry is None or self.gradients[channel] is None:
            return
        from ..Grid.point_group import symmetrize_vector

        group = getattr(density_symmetry, "group", density_symmetry)
        public = np.stack(self._flux_to_public(self.gradients[channel]))
        averaged = symmetrize_vector(public, group)
        self.gradients[channel] = self._flux_to_internal(tuple(averaged))

    def gradient_keywords(self, channel: int = 0) -> dict:
        """``density_gradient`` of an unpolarized ``evaluate_xc`` in the orbital form, else nothing."""

        if not self.orbital_gradient:
            return {}
        return {"density_gradient": self.gradients[channel]}

    def spin_gradient_keywords(self) -> dict:
        """``density_gradients`` of ``evaluate_spin_xc`` in the orbital form, else nothing."""

        if not self.orbital_gradient:
            return {}
        return {"density_gradients": tuple(self.gradients)}

    # ----- mixing -------------------------------------------------------------------

    def _block_count(self) -> int:
        return (5 if self.orbital_gradient else 2) * self.channels

    def mixer(self, build_mixer, settings):
        """The mixer of the loop: one over ``(V, vtau[, W])`` joined, or the plain one (``output``)."""

        self.parameter = float(getattr(settings, "parameter", 0.3))
        if self.mixing in ("output", "linear"):
            return build_mixer(settings)
        if self.adapter is None:
            if settings.method == "anderson" and meta_gga_anderson() == "weighted":
                return DensityWeightedAnderson(settings)
            return build_mixer(settings)
        block_mixer = getattr(self.adapter, "block_mixer", None)
        if block_mixer is None:
            raise NotImplementedError(
                "this symmetry-reduced loop cannot mix the meta-GGA's vtau with its potential"
            )
        return block_mixer(settings, self._block_count())

    def _join(self, fields):
        if self.adapter is None:
            return np.concatenate([np.asarray(field, dtype=np.float64) for field in fields])
        return self.adapter.join_blocks(fields)

    def _split(self, values, count):
        if self.adapter is None:
            values = np.asarray(values, dtype=np.float64)
            return [values[index * self.size : (index + 1) * self.size].copy() for index in range(count)]
        return self.adapter.split_blocks(values, count)

    def _flat_flux(self, fluxes):
        return [component for flux in fluxes for component in flux]

    def mix(
        self,
        mixer,
        input_potentials,
        output_potentials,
        output_kinetic_potentials,
        *,
        iteration: int,
        join=None,
        split=None,
        densities=None,
    ):
        """Mix the potentials of the channels (and ``vtau``, ``W``); returns the mixed potentials.

        ``join``/``split`` are the loop's own joining of the potentials of the
        channels (spin, ``output`` mixing); the mixed ``vtau`` and ``W`` wait
        in :attr:`pending` until :meth:`advance`.  The output ``W`` of the
        iteration was taken by :meth:`energies`.
        """

        outputs = list(output_kinetic_potentials)
        self.outputs = outputs
        set_weights = getattr(mixer, "set_weights", None)
        if callable(set_weights) and densities is not None:
            # A density-weighted mixer (meta_gga_anderson()): the density of
            # each channel for its V, vtau and the components of W.
            wedge = (lambda field: np.asarray(field, dtype=np.float64)) if self.adapter is None else self.adapter.wedge_values
            fields = [wedge(density) for density in densities]
            weights = fields + fields
            if self.orbital_gradient:
                weights = fields + fields + [field for field in fields for _ in range(3)]
            set_weights(np.concatenate(weights))
        self.residual = max(
            self._residual_norm(output, given) for output, given in zip(outputs, self.input)
        )
        if self.mixing in ("output", "linear"):
            if self.channels == 1:
                mixed = [mixer.mix(input_potentials[0], output_potentials[0], iteration=iteration)]
            else:
                mixed = split(mixer.mix(join(input_potentials), join(output_potentials), iteration=iteration))
            if self.mixing == "output":
                self.pending = [output for output in outputs]
            else:
                self.pending = [self._linear(given, output) for output, given in zip(outputs, self.input)]
            return mixed
        inputs = list(input_potentials) + list(self.input)
        results = list(output_potentials) + outputs
        if self.orbital_gradient:
            inputs += self._flat_flux(self.input_flux)
            results += self._flat_flux(self.output_flux)
        parts = self._split(mixer.mix(self._join(inputs), self._join(results), iteration=iteration), self._block_count())
        channels = self.channels
        self.pending = parts[channels : 2 * channels]
        if self.orbital_gradient:
            flux = parts[2 * channels :]
            self.pending_flux = [tuple(flux[3 * channel : 3 * channel + 3]) for channel in range(channels)]
        return parts[:channels]

    def _linear(self, given, output):
        """``vtau_in + beta (vtau_out - vtau_in)`` with the mixing parameter."""

        beta = self.parameter
        if self.adapter is None:
            return np.asarray(given) + beta * (np.asarray(output) - np.asarray(given))
        wedge = self.adapter.wedge_values
        return self.adapter.field(wedge(given) + beta * (wedge(output) - wedge(given)))

    def advance(self) -> None:
        """The mixed ``vtau`` (and ``W``) become the input of the next iteration."""

        if self.pending is not None:
            self.input = list(self.pending)
            self.input_is_zero = False
            self.pending = None
        if self.pending_flux is not None:
            self.input_flux = list(self.pending_flux)
            self.pending_flux = None

    def _dot(self, left, right) -> float:
        if self.adapter is None:
            return float(np.dot(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)))
        wedge = self.adapter.wedge_values
        return float(np.dot(self.adapter.weights * wedge(left), wedge(right)))

    def _residual_norm(self, output, given) -> float:
        volume = float(self.system.grid.volume_element)
        if self.adapter is None:
            difference = np.asarray(output, dtype=np.float64) - np.asarray(given, dtype=np.float64)
            return float(np.sqrt(volume * np.dot(difference, difference)))
        wedge = self.adapter.wedge_values
        difference = wedge(output) - wedge(given)
        return float(np.sqrt(volume * np.dot(self.adapter.weights * difference, difference)))

    # ----- energies -------------------------------------------------------------------

    def double_counting(self, taus) -> float:
        """``2 h^3 sum_s [vtau_in,s tau_s + W_in,s . g_s]`` (Ry) of the orbitals of this iteration."""

        if self.input_is_zero:
            return 0.0
        volume = float(self.system.grid.volume_element)
        total = 0.0
        for channel, (potential, tau) in enumerate(zip(self.input, taus)):
            total += 2.0 * volume * self._dot(potential, tau)
            if self.orbital_gradient:
                for flux, gradient in zip(self.input_flux[channel], self.gradients[channel]):
                    total += 2.0 * volume * self._dot(flux, gradient)
        return total

    def energies(self, energies, taus, clamped: int = 0, xc=None):
        """``energies`` with the double counting of the orbital terms removed (and ``W_out`` of ``xc`` taken)."""

        self.taus = list(taus)
        self.clamped = int(clamped)
        if self.orbital_gradient:
            if xc is None:
                raise ValueError("the orbital form needs the XC result of the iteration")
            fluxes = getattr(xc, "gradient_fluxes", None)
            fluxes = (xc.gradient_flux,) if fluxes is None else fluxes
            self.output_flux = [tuple(flux) for flux in fluxes]
            self.core_potential = getattr(xc, "core_potential", None)
        counted = self.double_counting(taus)
        fields = {name: getattr(energies, name) for name in (
            "eigenvalue", "hartree", "integral_vxc_rho", "exchange_correlation",
            "electron_ion", "ion_ion", "electronic", "total",
        )}
        fields["electronic"] = float(energies.electronic) - counted
        fields["total"] = float(energies.total) - counted
        return MetaGGAEnergyBreakdown(**fields, integral_vtau_tau=counted)

    # ----- the record -----------------------------------------------------------------

    def record(self) -> MetaGGARecord:
        public = self._to_public
        outputs = self.outputs or [self._zero() for _ in range(self.channels)]
        following = self.pending if self.pending is not None else self.input
        following_flux = self.pending_flux if self.pending_flux is not None else self.input_flux
        taus = self.taus or [self._zero() for _ in range(self.channels)]
        core = self.core_potential
        return MetaGGARecord(
            kinetic_potentials=tuple(np.asarray(public(field)) for field in self.input),
            output_kinetic_potentials=tuple(np.asarray(public(field)) for field in outputs),
            next_kinetic_potentials=tuple(np.asarray(public(field)) for field in following),
            kinetic_energy_densities=tuple(np.asarray(public(field)) for field in taus),
            kinetic_residual=float(self.residual),
            fermi_hole_clamped=int(self.clamped),
            mixing=self.mixing,
            gradient="orbital" if self.orbital_gradient else "density",
            next_gradient_fluxes=(
                tuple(self._flux_to_public(flux) for flux in following_flux) if self.orbital_gradient else None
            ),
            core_potential=None if core is None else (core if self.adapter is None else public(core)),
        )


class DensityWeightedAnderson:
    """PARSEC's Anderson mixing (``anderson.f90``) with weighted inner products (:func:`meta_gga_anderson`).

    The weights are set before every step (:meth:`set_weights`); the
    candidate is ``anderson.f90``'s with the coefficients of the weighted
    least squares.
    """

    def __init__(self, settings) -> None:
        from ..Mixer import AndersonMixer

        self.settings = settings
        self._inner = AndersonMixer(settings)
        self.weights = None

    def set_weights(self, weights) -> None:
        self.weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)

    def reset(self) -> None:
        self._inner.reset()

    @property
    def safeguard_resets(self) -> int:
        return self._inner.safeguard_resets

    @property
    def safeguard_clips(self) -> int:
        return self._inner.safeguard_clips

    def mix(self, input_potential, output_potential, *, iteration=None):
        inner = self._inner
        x = np.asarray(input_potential, dtype=np.float64)
        residual = np.asarray(output_potential, dtype=np.float64) - x
        if iteration is None:
            iteration = inner._calls + 1
        if (iteration - 1) % self.settings.restart == 0:
            inner._clear_history()
        weights = np.ones_like(x) if self.weights is None else self.weights
        if not inner._residuals:
            mixed = x + self.settings.parameter * residual
        else:
            inputs = inner._inputs[-self.settings.memory :]
            residuals = inner._residuals[-self.settings.memory :]
            differences = np.column_stack([residual - previous for previous in residuals])
            gram = differences.T @ (weights[:, None] * differences)
            rhs = differences.T @ (weights * residual)
            if self.settings.regularization:
                # Mixing_Regularization, as anderson.f90's mixer and the sector mixer.
                scale = max(float(np.trace(gram)) / max(gram.shape[0], 1), 1.0)
                gram = gram + self.settings.regularization * scale * np.eye(gram.shape[0])
            try:
                coefficients = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
            average_input = x.copy()
            average_residual = residual.copy()
            for coefficient, previous_input, previous_residual in zip(coefficients, inputs, residuals):
                average_input += coefficient * (previous_input - x)
                average_residual += coefficient * (previous_residual - residual)
            mixed = average_input + self.settings.parameter * average_residual
        if self.settings.safeguard:
            # Mixing_Safeguard: the trust region of the plain mixer (plain norms,
            # the multiplicity-weighted norms of the sector mixer on the full grid).
            from ..Mixer.anderson import safeguard_anderson_candidate

            mixed, residual_norm, reset_history, clipped = safeguard_anderson_candidate(
                x, residual, mixed, self.settings, previous_residual_norm=inner._previous_residual_norm
            )
            if reset_history:
                inner._clear_history()
                inner.safeguard_resets += 1
            if clipped:
                inner.safeguard_clips += 1
            inner._previous_residual_norm = residual_norm
        inner._inputs.append(x.copy())
        inner._residuals.append(residual.copy())
        if len(inner._inputs) > self.settings.memory:
            inner._inputs.pop(0)
            inner._residuals.pop(0)
        inner._calls = iteration
        return mixed


def refuse_meta_gga_hooks(system, **hooks) -> None:
    """Refuse execution hooks that cannot carry a meta-GGA (none of the system's own)."""

    unsupported = [name for name, hook in hooks.items() if hook is not None]
    if unsupported and getattr(system, "meta_gga_device", None) is None:
        raise NotImplementedError(
            "a meta-GGA needs the kinetic-energy density of the orbitals; these execution hooks "
            "do not provide it: " + ", ".join(unsupported)
        )


__all__ = ["MetaGGALoop", "MetaGGARecord", "meta_gga_mixing", "refuse_meta_gga_hooks"]
