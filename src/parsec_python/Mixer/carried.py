"""Anderson mixing whose secant pairs outlive one SCF loop (the inner loops of a Fock loop).

The inner SCF loops of a hybrid functional (:mod:`parsec_python.SCF.hybrid`)
solve one fixed-point problem after another, ``F_k(V) = V_out[V; V_ACE^k] - V``,
with the exchange operator ``V_ACE^k`` fixed in loop ``k``.  The operator
enters the eigenproblem only, so from one loop to the next the map changes
mostly by an offset: in the linear regime ``F_{k+1}(V) ~ F_k(V) + c_k`` with
the same Jacobian ``J = dF/dV`` (charge sloshing, the Hartree and semilocal
kernels times the response of the orbitals).  A secant pair of loop ``k``,
``(dV, dF) = (V_{j+1} - V_j, F_{j+1} - F_j)``, therefore remains a secant pair
of loop ``k + 1`` (``dF = J dV`` there too), while an absolute pair
``(V_j, F_j)`` does not: its residual carries the old offset, and a residual
difference across the loop boundary would mostly measure the change of the
operator, not of the potential.

:class:`CarriedAndersonMixer` is the Anderson mixer of ``anderson.f90`` (with
``Mixing_Param``, ``Memory_Param`` and ``Restart_Mixing``) whose least
squares take, besides the differences ``F_k - F_i`` of the pairs of the
current loop, the first ``Memory_Param`` consecutive differences of the loop
before it ("seeds").  The first differences of a loop are the ones made at
large residuals, along the response to the operator change, which is the
direction of the next loop's first residual again (the Fock loop converges
linearly, each loop starting from a residual of the same shape, smaller).

At iteration ``k`` of a loop, with ``D_F = [F_k - F_i | dF_j]`` and
``D_V = [V_k - V_i | dV_j]``,

``c = argmin || F_k - D_F c ||_w``,
``V_{k+1} = (V_k - D_V c) + beta (F_k - D_F c)``

(``w`` the inner-product weights of the representation, the orbit
multiplicities of a symmetry wedge or 1).  Without seeds and columns this is
linear mixing, as the first call of ``anderson.f90``.  The least squares are
solved on unit-normalized columns by a pseudo-inverse of the Gram matrix
that drops directions below ``1e-12`` of its largest eigenvalue (seeds and
new pairs can be nearly parallel).  ``Restart_Mixing`` clears the pairs of the
current loop and the seeds, as PARSEC clears its history.

The fields of the loop pass through a codec: :class:`PlainVectors` for the
full grid (and the concatenated spin channels), :class:`ReducedVectors` for
the scalar fields of a symmetry wedge (one value per orbit, the
multiplicities as weights), or :class:`WeightedVectors` for the joined orbit
values of the two spin channels in the sectors.  :class:`RecordingMixer` wraps the
mixer of another loop (the semilocal start of the Fock loop), changes nothing
of its results and keeps its first differences as the seeds of the first
inner loop.

The fixed point is untouched: every step is an affine combination of
potentials of the current loop (and of potential differences), and the SCF
stop test is that of the loop.

Measured (A100, examples/0d_hybrid_functionals, 2026-11-06): together with
the adaptive inner criterion the carried history costs more inner iterations
than a fresh mixer per loop (N2, seven hybrids: 293 against 242; H2O: 291
against 231), and alone more than the baseline (N2: 440 against 359).  The
late secant pairs of a loop are made at the noise floor of the inner SCF
(weighted residuals 1e-6..1e-5), its early ones belong to a residual of
another shape; weighting the least squares by the density (the metric of
the stop test) did not help either (H2 B3LYP: 23 iterations in the first
inner loop against 8).  ``Exx_Mixer_History: restart`` is therefore the
default, and this mixer is the opt-in ``carry``.  Units Ry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import MixingSettings
from .anderson import safeguard_anderson_candidate

#: Directions of the normalized Gram matrix below this share of its largest eigenvalue are dropped.
GRAM_CUTOFF = 1.0e-12


class PlainVectors:
    """The fields of a loop as they are: full-grid vectors, unit weights."""

    weights = None

    @staticmethod
    def values(field) -> np.ndarray:
        return np.asarray(field, dtype=np.float64)

    @staticmethod
    def restore(template, values: np.ndarray):
        return values


class WeightedVectors:
    """Vectors the loop already joined, with inner-product weights.

    The spin loops of the symmetry sectors hand the mixer the orbit values of
    both channels joined (``SpinSectorReducer.join_channels``, up first); the
    weights are the orbit multiplicities of each half, as the default mixer of
    those loops (``SpinSectorReducer.spin_mixer``) weighs them.
    """

    def __init__(self, weights) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)

    def values(self, field) -> np.ndarray:
        values = np.asarray(field, dtype=np.float64)
        if values.shape != self.weights.shape:
            raise ValueError("the joined vector does not match the weights of the mixer")
        return values

    @staticmethod
    def restore(template, values: np.ndarray):
        return values


class ReducedVectors:
    """The scalar fields of a symmetry wedge (``SymmetrySCFReducer`` of the accelerated path).

    ``values`` gives one value per orbit, ``restore`` the field in the
    representation the loop handed in (a wedge field or a full-grid array),
    ``weights`` the orbit multiplicities.
    """

    def __init__(self, reducer) -> None:
        self.reducer = reducer
        self.weights = np.asarray(reducer.weights, dtype=np.float64)

    def values(self, field) -> np.ndarray:
        return np.array(self.reducer.wedge_values(field), dtype=np.float64)

    def restore(self, template, values: np.ndarray):
        if isinstance(template, np.ndarray):
            return self.reducer.expand_values(values)
        return self.reducer.field(values)


def _dot(left: np.ndarray, right: np.ndarray, weights: np.ndarray | None) -> float:
    return float(np.dot(left, right if weights is None else weights * right))


@dataclass
class CarriedAndersonMixer:
    """Anderson mixing of a sequence of SCF loops with carried secant pairs (see the module).

    One object serves every inner loop of a Fock loop
    (``SCFWarmStart.mixer``); a call with ``iteration=1`` starts the next
    loop, whose seeds are then the first differences of the loop before.
    """

    settings: MixingSettings = field(default_factory=MixingSettings)
    codec: object = field(default_factory=PlainVectors)
    _seeds: list = field(default_factory=list, init=False)
    _first: list = field(default_factory=list, init=False)
    _inputs: list = field(default_factory=list, init=False)
    _residuals: list = field(default_factory=list, init=False)
    _calls: int = field(default=0, init=False)
    _loops: int = field(default=0, init=False)
    _previous_residual_norm: float | None = field(default=None, init=False)
    safeguard_resets: int = field(default=0, init=False)
    safeguard_clips: int = field(default=0, init=False)

    @property
    def seed_count(self) -> int:
        """Secant pairs carried into the current loop."""

        return len(self._seeds)

    def seed(self, pairs) -> None:
        """Take ``(dV, dF)`` pairs (codec values) as the seeds of the next loop."""

        self._first = [(np.array(dv, dtype=np.float64), np.array(df, dtype=np.float64)) for dv, df in pairs]
        self._loops = max(self._loops, 1)

    def reset(self) -> None:
        self._seeds.clear()
        self._first.clear()
        self._clear_loop()
        self._calls = 0
        self._loops = 0
        self.safeguard_resets = 0
        self.safeguard_clips = 0

    def _clear_loop(self) -> None:
        self._inputs.clear()
        self._residuals.clear()
        self._previous_residual_norm = None

    def _start_loop(self) -> None:
        if self._loops:
            # The first differences of the loop that ended, then older seeds.
            self._seeds = (self._first + self._seeds)[: self.settings.memory]
        self._first = []
        self._clear_loop()
        self._loops += 1

    def mix(self, input_potential, output_potential, *, iteration: int | None = None):
        if iteration is None:
            iteration = self._calls + 1
        if iteration < 1:
            raise ValueError("SCF iteration numbers start at one")
        codec = self.codec
        weights = codec.weights
        x = codec.values(input_potential)
        residual = codec.values(output_potential) - x
        if x.ndim != 1 or residual.shape != x.shape:
            raise ValueError("input and output potentials must be identical one-dimensional vectors")
        if iteration == 1:
            self._start_loop()
        elif (iteration - 1) % self.settings.restart == 0:
            # Restart_Mixing: the history of the loop and the seeds are stale.
            self._clear_loop()
            self._seeds = []
        if self._inputs and len(self._first) < self.settings.memory:
            self._first.append((x - self._inputs[-1], residual - self._residuals[-1]))

        recent = list(zip(self._inputs[-self.settings.memory :], self._residuals[-self.settings.memory :]))
        columns_f = [residual - previous for _, previous in recent] + [df for _, df in self._seeds]
        columns_v = [x - previous for previous, _ in recent] + [dv for dv, _ in self._seeds]
        if not columns_f:
            mixed = x + self.settings.parameter * residual
        else:
            coefficients = self._coefficients(columns_f, residual, weights)
            average_input = x.copy()
            average_residual = residual.copy()
            for coefficient, column_v, column_f in zip(coefficients, columns_v, columns_f):
                if coefficient:
                    average_input -= coefficient * column_v
                    average_residual -= coefficient * column_f
            mixed = average_input + self.settings.parameter * average_residual

        if self.settings.safeguard:
            mixed, residual_norm, reset_history, clipped = safeguard_anderson_candidate(
                x,
                residual,
                mixed,
                self.settings,
                previous_residual_norm=self._previous_residual_norm,
                weights=weights,
            )
            if reset_history:
                self._clear_loop()
                self._seeds = []
                self.safeguard_resets += 1
            if clipped:
                self.safeguard_clips += 1
            self._previous_residual_norm = residual_norm

        self._inputs.append(x)
        self._residuals.append(residual)
        if len(self._inputs) > self.settings.memory + 1:
            # One pair more than the least squares read: the next consecutive difference.
            self._inputs.pop(0)
            self._residuals.pop(0)
        self._calls = iteration
        return codec.restore(input_potential, mixed)

    def _coefficients(self, columns, residual: np.ndarray, weights) -> np.ndarray:
        """``argmin || r - sum_j c_j d_j ||_w`` on unit columns, small directions dropped."""

        norms = np.array([np.sqrt(max(_dot(column, column, weights), 0.0)) for column in columns])
        usable = norms > 0.0
        count = len(columns)
        gram = np.zeros((count, count))
        rhs = np.zeros(count)
        for i in range(count):
            if not usable[i]:
                continue
            rhs[i] = _dot(columns[i], residual, weights) / norms[i]
            for j in range(i, count):
                if usable[j]:
                    gram[i, j] = gram[j, i] = _dot(columns[i], columns[j], weights) / (norms[i] * norms[j])
        if self.settings.regularization:
            scale = max(float(np.trace(gram)) / max(count, 1), 1.0)
            gram = gram + self.settings.regularization * scale * np.eye(count)
        values, vectors = np.linalg.eigh(gram)
        keep = values > GRAM_CUTOFF * max(float(values.max(initial=0.0)), 0.0)
        if not np.any(keep):
            return np.zeros(count)
        solution = vectors[:, keep] @ ((vectors[:, keep].T @ rhs) / values[keep])
        solution[~usable] = 0.0
        return np.where(usable, solution / np.where(norms > 0.0, norms, 1.0), 0.0)


class RecordingMixer:
    """Another loop's mixer, unchanged, that keeps the first differences of its loop.

    ``mix`` returns what ``inner.mix`` returns; the first ``count``
    consecutive ``(dV, dF)`` of the loop (codec values) are kept in
    ``pairs`` for :meth:`CarriedAndersonMixer.seed`.
    """

    def __init__(self, inner, codec, count: int) -> None:
        self.inner = inner
        self.codec = codec
        self.count = int(count)
        self.pairs: list = []
        self._last = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def mix(self, input_potential, output_potential, *, iteration: int | None = None):
        if len(self.pairs) < self.count:
            x = self.codec.values(input_potential)
            residual = self.codec.values(output_potential) - x
            if self._last is not None:
                self.pairs.append((x - self._last[0], residual - self._last[1]))
            self._last = (x, residual)
        else:
            self._last = None
        return self.inner.mix(input_potential, output_potential, iteration=iteration)


__all__ = [
    "CarriedAndersonMixer",
    "GRAM_CUTOFF",
    "PlainVectors",
    "RecordingMixer",
    "ReducedVectors",
    "WeightedVectors",
]
