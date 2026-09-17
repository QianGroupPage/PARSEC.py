"""Orbit-reduced SCF scalar algebra for an exact reflection symmetry.

The Kohn--Sham density and all scalar local potentials transform as the
totally symmetric representation.  One physical value per grid orbit is
therefore sufficient for residual norms, Anderson history, and real-space
energy integrals.  Orbit multiplicities reproduce the full-grid quadrature:

``sum_i f_i = sum_w m_w f_w``.

This reduction changes storage and summation topology only.  It does not
alter the density functional, mixer equation, convergence criterion, or
energy expression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from parsec_python.Mixer import ResidualMetrics
from parsec_python.Mixer.anderson import safeguard_anderson_candidate
from parsec_python.models import EnergyBreakdown, MixingSettings

from ..Symmetry.axis_reflection import AxisReflectionReduction


@dataclass(frozen=True)
class SymmetryScalarField:
    """One physical scalar value per exact real-space symmetry orbit.

    Wavefunctions use normalized wedge coordinates, whereas density and local
    potentials are physical point values.  Keeping that distinction in the
    type prevents accidental ``sqrt(m)`` factors while allowing the entire
    nonlinear SCF path to avoid repeated full-grid expansion.
    """

    reduction: AxisReflectionReduction
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.values, dtype=np.float64)
        if values.shape != (self.reduction.wedge_size,):
            raise ValueError("symmetry scalar field does not match the wedge")
        if not np.all(np.isfinite(values)):
            raise ValueError("symmetry scalar field contains nonfinite values")
        object.__setattr__(self, "values", values)

    def copy(self) -> "SymmetryScalarField":
        return SymmetryScalarField(self.reduction, self.values.copy())

    def _binary(self, other, operation) -> "SymmetryScalarField":
        if not isinstance(other, SymmetryScalarField):
            return NotImplemented
        if other.reduction is not self.reduction:
            raise ValueError("symmetry scalar fields use different orbit maps")
        return SymmetryScalarField(
            self.reduction, operation(self.values, other.values)
        )

    def __add__(self, other):
        return self._binary(other, np.add)

    def __sub__(self, other):
        return self._binary(other, np.subtract)

    def __neg__(self) -> "SymmetryScalarField":
        return SymmetryScalarField(self.reduction, -self.values)


@dataclass(frozen=True)
class SymmetrySCFReducer:
    """Evaluate invariant scalar-field operations on one value per orbit."""

    reduction: AxisReflectionReduction

    def field(self, values: np.ndarray) -> SymmetryScalarField:
        """Construct a validated physical wedge field."""

        return SymmetryScalarField(self.reduction, values)

    def from_full(self, values: np.ndarray) -> SymmetryScalarField:
        """Orbit-average one full invariant field into persistent storage."""

        return self.field(self.wedge_values(values))

    def to_full(self, values) -> np.ndarray:
        """Materialize a public full-grid result only at an API boundary."""

        return self.expand_values(self.wedge_values(values))

    def minimum(self, values) -> float:
        return float(np.min(self.wedge_values(values)))

    def maximum(self, values) -> float:
        return float(np.max(self.wedge_values(values)))

    def wedge_values(self, values) -> np.ndarray:
        """Orbit-average a physical scalar field (without U normalization)."""

        if isinstance(values, SymmetryScalarField):
            if values.reduction is not self.reduction:
                raise ValueError("scalar field uses a different symmetry map")
            return values.values
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.reduction.full_size,):
            raise ValueError("scalar field does not match the symmetry grid")
        sums = np.bincount(
            self.reduction.full_to_wedge,
            weights=array,
            minlength=self.reduction.wedge_size,
        )
        return sums / self.reduction.multiplicities

    def expand_values(self, wedge_values) -> np.ndarray:
        """Expand physical orbit values without wavefunction normalization."""

        values = self.wedge_values(wedge_values) if isinstance(
            wedge_values, SymmetryScalarField
        ) else np.asarray(wedge_values, dtype=np.float64)
        if values.shape != (self.reduction.wedge_size,):
            raise ValueError("wedge field does not match the symmetry map")
        return np.ascontiguousarray(values[self.reduction.full_to_wedge])

    def weighted_dot(self, left: np.ndarray, right: np.ndarray) -> float:
        """Return the exact full-grid dot product for invariant fields."""

        left_wedge = self.wedge_values(left)
        right_wedge = self.wedge_values(right)
        return float(
            np.dot(
                self.reduction.multiplicities * left_wedge,
                right_wedge,
            )
        )

    def potential_residual_metrics(
        self,
        input_potential: np.ndarray,
        output_potential: np.ndarray,
        density: np.ndarray,
        volume_element: float,
        electron_count: float,
    ) -> ResidualMetrics:
        """Evaluate PARSEC SRE norms with orbit-multiplicity quadrature."""

        if volume_element <= 0.0 or electron_count <= 0.0:
            raise ValueError("electron count and volume element must be positive")
        input_wedge = self.wedge_values(input_potential)
        output_wedge = self.wedge_values(output_potential)
        density_wedge = self.wedge_values(density)
        residual_wedge = output_wedge - input_wedge
        multiplicities = self.reduction.multiplicities
        residual_squared = residual_wedge * residual_wedge
        plain_squared = volume_element * np.dot(
            multiplicities, residual_squared
        )
        weighted_squared = (
            volume_element
            * np.dot(multiplicities * density_wedge, residual_squared)
            / electron_count
        )
        return ResidualMetrics(
            weighted=float(np.sqrt(max(float(weighted_squared), 0.0))),
            plain=float(np.sqrt(max(float(plain_squared), 0.0))),
            residual=self.expand_values(residual_wedge),
        )

    def mixer(self, settings: MixingSettings) -> "SymmetryAndersonMixer":
        return SymmetryAndersonMixer(self, settings)

    def total_energy(
        self,
        eigenvalues: np.ndarray,
        occupations: np.ndarray,
        density: np.ndarray,
        input_effective_potential: np.ndarray,
        ionic_potential: np.ndarray,
        output_hartree_potential: np.ndarray,
        output_xc_potential: np.ndarray,
        exchange_correlation_energy: float,
        ion_ion_energy: float,
        volume_element: float,
    ) -> EnergyBreakdown:
        """Evaluate the unchanged PARSEC energy using weighted wedge dots."""

        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        occupations = np.asarray(occupations, dtype=np.float64)
        if eigenvalues.shape != occupations.shape:
            raise ValueError("eigenvalues and occupations must have the same shape")
        density_wedge = self.wedge_values(density)
        multiplicities = self.reduction.multiplicities

        def density_integral(field: np.ndarray) -> float:
            field_wedge = self.wedge_values(field)
            return float(
                volume_element
                * np.dot(multiplicities * density_wedge, field_wedge)
            )

        band_energy = float(2.0 * np.dot(occupations, eigenvalues))
        old_hxc_integral = float(
            volume_element
            * np.dot(
                multiplicities * density_wedge,
                self.wedge_values(input_effective_potential)
                - self.wedge_values(ionic_potential),
            )
        )
        hartree_integral = density_integral(output_hartree_potential)
        vxc_integral = density_integral(output_xc_potential)
        electron_ion = density_integral(ionic_potential)
        electronic = float(
            band_energy
            - old_hxc_integral
            + 0.5 * hartree_integral
            + exchange_correlation_energy
        )
        return EnergyBreakdown(
            eigenvalue=band_energy,
            hartree=0.5 * hartree_integral,
            integral_vxc_rho=vxc_integral,
            exchange_correlation=float(exchange_correlation_energy),
            electron_ion=electron_ion,
            ion_ion=float(ion_ion_energy),
            electronic=electronic,
            total=electronic + float(ion_ion_energy),
        )


@dataclass
class SymmetryAndersonMixer:
    """PARSEC Anderson mixing with multiplicity-weighted wedge inner products."""

    reducer: SymmetrySCFReducer
    settings: MixingSettings = field(default_factory=MixingSettings)
    _inputs: list[np.ndarray] = field(default_factory=list, init=False)
    _residuals: list[np.ndarray] = field(default_factory=list, init=False)
    _calls: int = field(default=0, init=False)
    _previous_residual_norm: float | None = field(default=None, init=False)
    safeguard_resets: int = field(default=0, init=False)
    safeguard_clips: int = field(default=0, init=False)

    def _clear_history(self) -> None:
        self._inputs.clear()
        self._residuals.clear()

    def reset(self) -> None:
        self._clear_history()
        self._calls = 0
        self._previous_residual_norm = None
        self.safeguard_resets = 0
        self.safeguard_clips = 0

    def mix(
        self,
        input_potential: np.ndarray,
        output_potential: np.ndarray,
        *,
        iteration: int | None = None,
    ) -> np.ndarray:
        input_wedge = self.reducer.wedge_values(input_potential)
        output_wedge = self.reducer.wedge_values(output_potential)
        if iteration is None:
            iteration = self._calls + 1
        if iteration < 1:
            raise ValueError("SCF iteration numbers start at one")
        if (iteration - 1) % self.settings.restart == 0:
            self._clear_history()

        residual = output_wedge - input_wedge
        if not self._residuals:
            mixed = input_wedge + self.settings.parameter * residual
        else:
            previous_inputs = self._inputs[-self.settings.memory :]
            previous_residuals = self._residuals[-self.settings.memory :]
            differences = np.column_stack(
                [residual - previous for previous in previous_residuals]
            )
            weighted_differences = (
                self.reducer.reduction.multiplicities[:, None] * differences
            )
            gram = differences.T @ weighted_differences
            rhs = differences.T @ (
                self.reducer.reduction.multiplicities * residual
            )
            if self.settings.regularization:
                scale = max(float(np.trace(gram)) / max(gram.shape[0], 1), 1.0)
                gram = gram + self.settings.regularization * scale * np.eye(
                    gram.shape[0]
                )
            try:
                coefficients = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
            average_input = input_wedge.copy()
            average_residual = residual.copy()
            for coefficient, previous_input, previous_residual in zip(
                coefficients, previous_inputs, previous_residuals
            ):
                average_input += coefficient * (previous_input - input_wedge)
                average_residual += coefficient * (previous_residual - residual)
            mixed = average_input + self.settings.parameter * average_residual

        if self.settings.safeguard:
            mixed, residual_norm, reset_history, clipped = (
                safeguard_anderson_candidate(
                    input_wedge,
                    residual,
                    mixed,
                    self.settings,
                    previous_residual_norm=self._previous_residual_norm,
                    weights=self.reducer.reduction.multiplicities,
                )
            )
            if reset_history:
                self._clear_history()
                self.safeguard_resets += 1
            if clipped:
                self.safeguard_clips += 1
            self._previous_residual_norm = residual_norm

        self._inputs.append(input_wedge.copy())
        self._residuals.append(residual.copy())
        if len(self._inputs) > self.settings.memory:
            self._inputs.pop(0)
            self._residuals.pop(0)
        self._calls = iteration
        if isinstance(input_potential, SymmetryScalarField):
            return self.reducer.field(mixed)
        return self.reducer.expand_values(mixed)


__all__ = [
    "SymmetryAndersonMixer",
    "SymmetrySCFReducer",
    "SymmetryScalarField",
]
