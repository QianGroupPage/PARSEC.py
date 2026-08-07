"""PARSEC-style effective-potential residuals and Anderson mixing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import MixingSettings


@dataclass(frozen=True)
class ResidualMetrics:
    weighted: float
    plain: float
    residual: np.ndarray


def potential_residual_metrics(
    input_potential: np.ndarray,
    output_potential: np.ndarray,
    density: np.ndarray,
    volume_element: float,
    electron_count: float,
) -> ResidualMetrics:
    """Evaluate PARSEC's weighted and plain SCF residual norms."""
    input_potential = np.asarray(input_potential, dtype=float)
    output_potential = np.asarray(output_potential, dtype=float)
    density = np.asarray(density, dtype=float)
    if input_potential.shape != output_potential.shape or density.shape != input_potential.shape:
        raise ValueError("potential and density arrays must have identical shapes")
    if electron_count <= 0 or volume_element <= 0:
        raise ValueError("electron count and volume element must be positive")
    residual = output_potential - input_potential
    plain = float(np.sqrt(volume_element * np.dot(residual, residual)))
    weighted = float(
        np.sqrt(
            volume_element
            * np.dot(density, residual * residual)
            / electron_count
        )
    )
    return ResidualMetrics(weighted=weighted, plain=plain, residual=residual)


@dataclass
class AndersonMixer:
    """Potential mixer matching ``anderson.f90`` for a scalar calculation."""

    settings: MixingSettings = field(default_factory=MixingSettings)
    _inputs: list[np.ndarray] = field(default_factory=list, init=False)
    _residuals: list[np.ndarray] = field(default_factory=list, init=False)
    _calls: int = field(default=0, init=False)

    def _clear_history(self) -> None:
        self._inputs.clear()
        self._residuals.clear()

    def reset(self) -> None:
        self._clear_history()
        self._calls = 0

    def mix(
        self,
        input_potential: np.ndarray,
        output_potential: np.ndarray,
        *,
        iteration: int | None = None,
    ) -> np.ndarray:
        input_potential = np.asarray(input_potential, dtype=float)
        output_potential = np.asarray(output_potential, dtype=float)
        if (
            input_potential.ndim != 1
            or input_potential.shape != output_potential.shape
        ):
            raise ValueError(
                "input and output potentials must be identical one-dimensional vectors"
            )

        if iteration is None:
            iteration = self._calls + 1
        if iteration < 1:
            raise ValueError("SCF iteration numbers start at one")
        if (iteration - 1) % self.settings.restart == 0:
            self._clear_history()

        residual = output_potential - input_potential
        if not self._residuals:
            mixed = input_potential + self.settings.parameter * residual
        else:
            previous_inputs = self._inputs[-self.settings.memory :]
            previous_residuals = self._residuals[-self.settings.memory :]
            residual_differences = np.column_stack(
                [residual - previous for previous in previous_residuals]
            )
            gram = residual_differences.T @ residual_differences
            rhs = residual_differences.T @ residual
            if self.settings.regularization:
                scale = max(float(np.trace(gram)) / max(gram.shape[0], 1), 1.0)
                gram = gram + self.settings.regularization * scale * np.eye(gram.shape[0])
            try:
                coefficients = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]

            average_input = input_potential.copy()
            average_residual = residual.copy()
            for coefficient, previous_input, previous_residual in zip(
                coefficients, previous_inputs, previous_residuals
            ):
                average_input += coefficient * (previous_input - input_potential)
                average_residual += coefficient * (previous_residual - residual)
            mixed = average_input + self.settings.parameter * average_residual

        self._inputs.append(input_potential.copy())
        self._residuals.append(residual.copy())
        if len(self._inputs) > self.settings.memory:
            self._inputs.pop(0)
            self._residuals.pop(0)
        self._calls = iteration
        return mixed


__all__ = ["AndersonMixer", "ResidualMetrics", "potential_residual_metrics"]
