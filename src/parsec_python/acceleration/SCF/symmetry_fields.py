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
from functools import cached_property
import os

import numpy as np

from parsec_python.Mixer.anderson import safeguard_anderson_candidate
from parsec_python.models import EnergyBreakdown, MixingSettings

from ..Symmetry.axis_reflection import AxisReflectionReduction

# Rows that one pass of the buffered Anderson step works on: the six vectors
# it touches stay in a cache of a megabyte while the history streams by.
_BLOCK_ROWS = 16384


def scf_buffers_requested() -> bool:
    """``PARSEC_SYMMETRY_SCF_BUFFERS``: on unless 0, false, no or off.

    On, the weighted wedge sums read the orbit multiplicities as float64, an
    energy forms the weighted density once for its four integrals, and the
    Anderson step works block by block in arrays the mixer keeps.  Off, every
    integral and every term of the step allocates its own arrays, as before.
    Each floating-point operation has the same operands in the same order
    either way, and the dense products receive arrays of the same shape and
    layout, so every norm, energy and mixed potential is the same bit for bit.
    """

    return os.environ.get("PARSEC_SYMMETRY_SCF_BUFFERS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


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
class SymmetryResidualMetrics:
    """PARSEC SRE norms with the residual kept on one value per orbit.

    This mirrors :class:`parsec_python.Mixer.ResidualMetrics`.  The SCF loop
    reads only the two norms, so the full-grid ``residual`` of that interface
    is expanded when it is read instead of at every SCF step.
    """

    weighted: float
    plain: float
    reducer: "SymmetrySCFReducer"
    wedge_residual: np.ndarray

    @property
    def residual(self) -> np.ndarray:
        """Return ``V_out - V_in`` on the full grid."""

        return self.reducer.expand_values(self.wedge_residual)


@dataclass(frozen=True)
class SymmetrySCFReducer:
    """Evaluate invariant scalar-field operations on one value per orbit."""

    reduction: AxisReflectionReduction

    @cached_property
    def weights(self) -> np.ndarray:
        """The orbit multiplicities as float64, read-only.

        A product of the integer multiplicities with a float64 field converts
        each of them to this value first, so either operand gives the same
        product; this one needs no conversion per call.
        """

        weights = np.array(self.reduction.multiplicities, dtype=np.float64)
        weights.setflags(write=False)
        return weights

    def _quadrature_weights(self) -> np.ndarray:
        if scf_buffers_requested():
            return self.weights
        return self.reduction.multiplicities

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
                self._quadrature_weights() * left_wedge,
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
    ) -> SymmetryResidualMetrics:
        """Evaluate PARSEC SRE norms with orbit-multiplicity quadrature."""

        if volume_element <= 0.0 or electron_count <= 0.0:
            raise ValueError("electron count and volume element must be positive")
        input_wedge = self.wedge_values(input_potential)
        output_wedge = self.wedge_values(output_potential)
        density_wedge = self.wedge_values(density)
        residual_wedge = output_wedge - input_wedge
        multiplicities = self._quadrature_weights()
        residual_squared = residual_wedge * residual_wedge
        plain_squared = volume_element * np.dot(
            multiplicities, residual_squared
        )
        weighted_squared = (
            volume_element
            * np.dot(multiplicities * density_wedge, residual_squared)
            / electron_count
        )
        return SymmetryResidualMetrics(
            weighted=float(np.sqrt(max(float(weighted_squared), 0.0))),
            plain=float(np.sqrt(max(float(plain_squared), 0.0))),
            reducer=self,
            wedge_residual=residual_wedge,
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
        alpha_z_energy: float = 0.0,
    ) -> EnergyBreakdown:
        """Evaluate the unchanged PARSEC energy using weighted wedge dots.

        ``alpha_z_energy`` joins the band energy as in the reference
        ``total_energy``, which the SCF loop calls with the same arguments.
        It is zero for an isolated system.
        """

        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        occupations = np.asarray(occupations, dtype=np.float64)
        if eigenvalues.shape != occupations.shape:
            raise ValueError("eigenvalues and occupations must have the same shape")
        density_wedge = self.wedge_values(density)
        multiplicities = self._quadrature_weights()
        # One product for the four integrals, or one each as before.
        shared = (
            multiplicities * density_wedge if scf_buffers_requested() else None
        )

        def weighted_density() -> np.ndarray:
            return multiplicities * density_wedge if shared is None else shared

        def density_integral(field: np.ndarray) -> float:
            field_wedge = self.wedge_values(field)
            return float(
                volume_element * np.dot(weighted_density(), field_wedge)
            )

        band_energy = float(2.0 * np.dot(occupations, eigenvalues)) + float(
            alpha_z_energy
        )
        old_hxc_integral = float(
            volume_element
            * np.dot(
                weighted_density(),
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
    # Arrays of the buffered step, by name, and history arrays that left the
    # history and take the next residual and the next copy of the input.
    _buffers: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    _spare: list[np.ndarray] = field(default_factory=list, init=False)

    def _clear_history(self) -> None:
        self._inputs.clear()
        self._residuals.clear()

    def reset(self) -> None:
        self._clear_history()
        self._calls = 0
        self._previous_residual_norm = None
        self.safeguard_resets = 0
        self.safeguard_clips = 0
        self._buffers.clear()
        self._spare.clear()

    def _buffer(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        """An array of the mixer with this shape, contents undefined."""

        array = self._buffers.get(name)
        if array is None or array.shape != shape:
            array = self._buffers[name] = np.empty(shape, dtype=np.float64)
        return array

    def _recycled(self, shape: tuple[int, ...]) -> np.ndarray | None:
        """A former history array of this shape to write into, if one is left."""

        while self._spare:
            array = self._spare.pop()
            if array.shape == shape:
                return array
        return None

    def _weighted_differences(
        self, residual: np.ndarray, previous_residuals: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Residual differences, their weighted copy and the weighted residual.

        The two matrices have one C-ordered row per wedge value and one
        column per history entry, as ``column_stack`` and the product with
        the multiplicities gave them, so the Gram matrix and the right-hand
        side come from the same dense products of the same arrays.
        """

        weights = self.reducer.weights
        shape = (residual.size, len(previous_residuals))
        differences = self._buffer("differences", shape)
        weighted = self._buffer("weighted_differences", shape)
        for start in range(0, residual.size, _BLOCK_ROWS):
            rows = slice(start, start + _BLOCK_ROWS)
            for column, previous in enumerate(previous_residuals):
                np.subtract(
                    residual[rows], previous[rows], out=differences[rows, column]
                )
                np.multiply(
                    weights[rows],
                    differences[rows, column],
                    out=weighted[rows, column],
                )
        weighted_residual = self._buffer("weighted", residual.shape)
        np.multiply(weights, residual, out=weighted_residual)
        return differences, weighted, weighted_residual

    def _averaged(
        self,
        input_wedge: np.ndarray,
        residual: np.ndarray,
        coefficients: np.ndarray,
        previous_inputs: list[np.ndarray],
        previous_residuals: list[np.ndarray],
    ) -> np.ndarray:
        """The Anderson candidate from the history, in a new array.

        Every value takes the additions of the unbuffered loop in its order,
        one block of rows at a time.
        """

        mixed = input_wedge.copy()
        block = (min(_BLOCK_ROWS, residual.size),)
        term_block = self._buffer("term", block)
        average_block = self._buffer("average_residual", block)
        for start in range(0, residual.size, _BLOCK_ROWS):
            rows = slice(start, start + _BLOCK_ROWS)
            current_input = input_wedge[rows]
            current_residual = residual[rows]
            average_input = mixed[rows]
            term = term_block[: current_residual.size]
            average_residual = average_block[: current_residual.size]
            np.copyto(average_residual, current_residual)
            for coefficient, previous_input, previous_residual in zip(
                coefficients, previous_inputs, previous_residuals
            ):
                np.subtract(previous_input[rows], current_input, out=term)
                np.multiply(coefficient, term, out=term)
                np.add(average_input, term, out=average_input)
                np.subtract(previous_residual[rows], current_residual, out=term)
                np.multiply(coefficient, term, out=term)
                np.add(average_residual, term, out=average_residual)
            np.multiply(
                self.settings.parameter, average_residual, out=average_residual
            )
            np.add(average_input, average_residual, out=average_input)
        return mixed

    def _safeguarded(
        self,
        input_wedge: np.ndarray,
        residual: np.ndarray,
        mixed: np.ndarray,
        weighted_residual: np.ndarray | None,
    ) -> tuple[np.ndarray, float, bool, bool]:
        """:func:`safeguard_anderson_candidate` with the mixer's arrays.

        ``mixed`` belongs to the mixer and is overwritten where the reference
        function returns a new candidate.  ``weighted_residual`` is the
        product of the weights with the residual where the caller has it.
        """

        settings = self.settings
        weights = self.reducer.weights
        weighted = self._buffer("weighted", residual.shape)
        if weighted_residual is not weighted:
            np.multiply(weights, residual, out=weighted)
        residual_norm = float(np.sqrt(np.dot(residual, weighted)))
        reset_history = bool(
            self._previous_residual_norm is not None
            and residual_norm
            > settings.growth_trigger * self._previous_residual_norm
        )
        if reset_history:
            np.multiply(settings.parameter * settings.backoff, residual, out=mixed)
            np.add(input_wedge, mixed, out=mixed)
        step = self._buffer("step", residual.shape)
        np.subtract(mixed, input_wedge, out=step)
        maximum_step_norm = settings.step_limit * settings.parameter * residual_norm
        np.multiply(weights, step, out=weighted)
        step_norm = float(np.sqrt(np.dot(step, weighted)))
        clipped = bool(step_norm > maximum_step_norm > 0.0)
        if clipped:
            np.multiply(step, maximum_step_norm / step_norm, out=step)
            np.add(input_wedge, step, out=mixed)
        return mixed, residual_norm, reset_history, clipped

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

        buffered = scf_buffers_requested()
        weighted_residual = None
        if buffered:
            # The history keeps this array itself; no one else holds it.
            residual = np.subtract(
                output_wedge, input_wedge, out=self._recycled(input_wedge.shape)
            )
        else:
            residual = output_wedge - input_wedge
        if not self._residuals:
            mixed = input_wedge + self.settings.parameter * residual
        else:
            previous_inputs = self._inputs[-self.settings.memory :]
            previous_residuals = self._residuals[-self.settings.memory :]
            if buffered:
                differences, weighted_differences, weighted_residual = (
                    self._weighted_differences(residual, previous_residuals)
                )
                gram = differences.T @ weighted_differences
                rhs = differences.T @ weighted_residual
            else:
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
            if buffered:
                mixed = self._averaged(
                    input_wedge,
                    residual,
                    coefficients,
                    previous_inputs,
                    previous_residuals,
                )
            else:
                average_input = input_wedge.copy()
                average_residual = residual.copy()
                for coefficient, previous_input, previous_residual in zip(
                    coefficients, previous_inputs, previous_residuals
                ):
                    average_input += coefficient * (previous_input - input_wedge)
                    average_residual += coefficient * (
                        previous_residual - residual
                    )
                mixed = average_input + self.settings.parameter * average_residual

        if self.settings.safeguard:
            mixed, residual_norm, reset_history, clipped = (
                self._safeguarded(input_wedge, residual, mixed, weighted_residual)
                if buffered
                else safeguard_anderson_candidate(
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

        if buffered:
            kept_input = self._recycled(input_wedge.shape)
            if kept_input is None:
                kept_input = np.empty_like(input_wedge)
            np.copyto(kept_input, input_wedge)
            self._inputs.append(kept_input)
            self._residuals.append(residual)
        else:
            self._inputs.append(input_wedge.copy())
            self._residuals.append(residual.copy())
        if len(self._inputs) > self.settings.memory:
            left = [self._inputs.pop(0), self._residuals.pop(0)]
            self._spare = left if buffered else []
        self._calls = iteration
        if isinstance(input_potential, SymmetryScalarField):
            return self.reducer.field(mixed)
        return self.reducer.expand_values(mixed)


__all__ = [
    "scf_buffers_requested",
    "SymmetryAndersonMixer",
    "SymmetryResidualMetrics",
    "SymmetrySCFReducer",
    "SymmetryScalarField",
]
