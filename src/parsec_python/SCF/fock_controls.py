"""How the Fock loop of a hybrid functional reaches its fixed point (not where it stops).

:func:`parsec_python.SCF.hybrid.run_hybrid_scf` stops when ``dexx <
Exx_Tolerance`` after an inner loop converged to ``Convergence_Criterion``.
This module holds the controls that change the path to that state:

``Exx_Inner_Criterion: adaptive`` -- an inexact inner loop while the Fock loop
is far from converged.  Inner loop ``k`` is solved with the operator of the
orbitals of loop ``k - 1``; its result only has to be accurate compared with
the change the next operator brings.  That change shows as the first
residual ``r_{k+1}`` of the next inner loop (weighted SRE, Ry), which falls
from loop to loop by the factor ``rho`` of the outer convergence.  Measured on
the 49 cases of examples/0d_hybrid_functionals (A100, 2026-11-06): ``rho`` is
0.06-0.25 for 20-25 % exact exchange and 0.25-0.47 for HF, and ``r_{k+1}``
is 0.05-0.4 ``sqrt(a_x dexx_k)``.  Loop ``k`` therefore stops at

``max(Convergence_Criterion, FORCING * rho * r_k)``

(the forcing term of an inexact Newton method, Eisenstat and Walker, SIAM J.
Sci. Comput. 17, 16 (1996): the inner error a tenth of the next outer step),
``r_k`` its own first residual, ``rho`` the ratio of the last two first
residuals (``FIRST_RATIO`` before two are known).  A loop that may be the last
or the one before it runs at ``Convergence_Criterion`` itself: the predicted
``dexx_k`` (the last ``dexx`` times the last ratio of two,
``FIRST_DEXX_RATIO`` before) below ``LAST_MARGIN * Exx_Tolerance`` (with a
margin of 3 an inexact loop before the last raised the last ``dexx`` by 12 %
in O2 PBE0 and cost an outer iteration there and in CO HF; with 30 the two
last loops are exact).  If an inexact loop ends with ``dexx`` below
the tolerance anyway, the loop is not converged and one more outer iteration
runs at ``Convergence_Criterion`` -- the final state is always an inner loop
at the input's criterion with ``dexx < Exx_Tolerance``, the stop rule of the
fixed loop.  This is Quantum ESPRESSO's adaptive threshold of its EXX loop
(``adaptive_thr``: ``tr2 = max(tr2_multi * dexx, tr2_final)``) in the residual
of a potential-mixing SCF.  As there (``conv_thr_init`` for the first SCF), the
semilocal start stops early too, at ``max(Convergence_Criterion,
START_CRITERION)``: the first inner loop starts from a residual of 3e-3..4e-2
Ry (the change from PBE to the hybrid; 20-25 % exact exchange to HF, the 49
cases), which a start converged below 1e-4 cannot change.  Measured on the
starts of those cases: below 1e-4 after 13-22 of their 20-44 iterations, the
PBE energy then within 1e-8..2.2e-6 Ry of the converged one -- the start's
energy is a start, not a converged PBE energy.  Only for closed shells: an
open shell can keep a soft rotation among degenerate partners (the occupied p
orbital of the O atom, 80 K, h 0.25, full grid: one of three p of the down
channel, its axis pinned by the cubic grid only), which the start loop still
turns when it stops at 1e-4 (44 of its 85 iterations); the exchange operator
of those orbitals then holds the axis, the Fock loop's steps along the
rotation are below Exx_Tolerance, and it ended 1.08e-7 Ry above the
axis-aligned state (Exx_Tolerance 1e-8; the old loop and the adaptive inner
criterion with a converged start reach it to 1e-11).  Symmetry sectors do not
forbid such a rotation in general (d_z2 and d_x2-y2 share a representation of
D2h, p partners one of Cs, C2 or Ci), so the rule does not look at the grid:
the start stops early only when it is spin-unpolarized
(:meth:`InnerCriterion.start_criterion`) and its occupations are integer
(largest ``min(f, 1 - f)`` below ``FRACTIONAL_START``,
:func:`fractional_start`); a spin-polarized start runs to
``Convergence_Criterion``, and an unpolarized one with a partly filled level
goes on from where it stopped to ``Convergence_Criterion``.

``Exx_Mixer_History: carry`` (opt-in; the default ``restart`` measured
faster, see :mod:`parsec_python.Mixer.carried`) -- one Anderson history
through the inner loops
(:class:`parsec_python.Mixer.carried.CarriedAndersonMixer`), seeded in the
first inner loop by the first secant pairs of the semilocal start
(:class:`~parsec_python.Mixer.carried.RecordingMixer`, which leaves the start
unchanged).  Only the Anderson ``Mixing_Method`` carries its history; the
Broyden and multisecant mixers start each inner loop afresh.

Units Ry.  Nothing here prints or opens files.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..Mixer.carried import (
    CarriedAndersonMixer,
    PlainVectors,
    RecordingMixer,
    ReducedVectors,
    WeightedVectors,
)

#: The inner residual a loop stops at, as a share of the predicted first residual of the next loop.
FORCING = 0.1
#: The weighted SRE (Ry) the semilocal start of an adaptive loop stops at (no lower than Convergence_Criterion).
START_CRITERION = 1.0e-4
#: Largest ``min(f, 1 - f)`` of the occupations of a start that may stop at START_CRITERION.
FRACTIONAL_START = 1.0e-3
#: The outer ratio of first residuals before two of them are known.
FIRST_RATIO = 0.1
#: The ratio of two dexx before two of them are known (20-25 % exact exchange: 0.005-0.02).
FIRST_DEXX_RATIO = 0.01
#: A loop whose predicted dexx is below this many Exx_Tolerance runs at Convergence_Criterion.
LAST_MARGIN = 30.0
#: Bounds of the measured ratio of first residuals.
RATIO_BOUNDS = (0.01, 0.9)


def selected_residual(record, use_plain: bool) -> float:
    """The residual an SCF iteration record was tested with (Ry)."""

    return float(record.plain_residual if use_plain else record.weighted_residual)


@dataclass
class InnerCriterion:
    """The stop rule of each inner loop (``Exx_Inner_Criterion``, see the module).

    ``final`` is the ``Convergence_Criterion`` of the run (Ry), ``tolerance``
    ``Exx_Tolerance``; :meth:`share` gives the ``relative_criterion`` of the
    warm start of an inner loop (``None``: the loop runs at ``final``),
    :meth:`observe` takes each finished loop and its ``dexx``.
    """

    adaptive: bool
    final: float
    tolerance: float
    use_plain: bool = False
    first_residuals: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def predicted_error(self) -> float | None:
        """``dexx`` expected after the next inner loop (``None`` before the first)."""

        if not self.errors:
            return None
        last = self.errors[-1]
        if len(self.errors) >= 2 and self.errors[-2] > 0.0:
            ratio = min(last / self.errors[-2], 1.0)
        else:
            ratio = FIRST_DEXX_RATIO
        return last * ratio

    def ratio(self) -> float:
        """The outer ratio of first residuals the next loop is expected to show."""

        if len(self.first_residuals) >= 2 and self.first_residuals[-2] > 0.0:
            low, high = RATIO_BOUNDS
            return min(max(self.first_residuals[-1] / self.first_residuals[-2], low), high)
        return FIRST_RATIO

    def start_criterion(self, spin_polarized: bool) -> float | None:
        """``convergence_criterion`` of the semilocal start, ``None`` for ``final``.

        Only a spin-unpolarized start stops early (see the module); one whose
        occupations then turn out fractional (:func:`fractional_start`) goes
        on to ``final``.
        """

        if not self.adaptive or spin_polarized or self.final >= START_CRITERION:
            return None
        return START_CRITERION

    def share(self) -> float | None:
        """``relative_criterion`` of the next inner loop, ``None`` for ``final``."""

        if not self.adaptive:
            return None
        predicted = self.predicted_error()
        if predicted is not None and predicted < LAST_MARGIN * self.tolerance:
            return None
        return FORCING * self.ratio()

    def observe(self, inner, error: float) -> None:
        history = getattr(inner, "history", ()) or ()
        if history:
            self.first_residuals.append(selected_residual(history[0], self.use_plain))
        self.errors.append(float(error))

    def threshold(self, inner, share: float | None) -> float:
        """The residual the loop ``inner`` stopped at (Ry)."""

        history = getattr(inner, "history", ()) or ()
        if share is None or not history:
            return self.final
        return max(self.final, share * selected_residual(history[0], self.use_plain))

    def tight(self, inner) -> bool:
        """Whether ``inner`` ended below ``Convergence_Criterion`` itself.

        The iteration judged is the last one of the SCF loop proper: an
        iteration of level passes after it (``SCFIteration.level_refinement``,
        the same ``V_in`` solved again) is not a step of the loop.
        """

        return bool(
            (last := last_loop_iteration(inner)) is not None
            and selected_residual(last, self.use_plain) < self.final
        )


def last_loop_iteration(inner):
    """The last SCF iteration of ``inner`` that is a step of the loop (not one of level passes), or ``None``."""

    for record in reversed(getattr(inner, "history", ()) or ()):
        if not getattr(record, "level_refinement", False):
            return record
    return None


def loop_iterations(inner) -> int:
    """The SCF iterations of ``inner`` without an iteration of level passes."""

    return sum(
        1 for record in getattr(inner, "history", ()) or () if not getattr(record, "level_refinement", False)
    )


def fock_loop_mixer(system, settings, hooks) -> CarriedAndersonMixer | None:
    """The carried mixer of the inner loops (``Exx_Mixer_History: carry``), ``None`` otherwise.

    ``hooks`` are the execution hooks of the run: a symmetry-reduced loop
    (``scalar_field_adapter`` with wedge values, unpolarized) mixes one value
    per orbit with the multiplicities as weights, the spin channels in the
    sectors their joined orbit values with the multiplicities of each half
    (the weights of the loop's own spin mixer), every other loop the
    full-grid fields (the spin loops their concatenated channels).
    """

    if settings.mixer_history != "carry" or system.input.mixing.method != "anderson":
        return None
    adapter = hooks.get("scalar_field_adapter")
    spin = bool(getattr(system.input.scf, "spin_polarized", False))
    if adapter is None:
        codec = PlainVectors()
    elif not spin and hasattr(adapter, "wedge_values"):
        codec = ReducedVectors(adapter)
    elif spin and hasattr(adapter, "join_channels"):
        # The spin channels in the sectors (Spin_Symmetry_Sectors): the joined
        # orbit values of both channels with the multiplicities of each half,
        # the weights of the loop's own spin mixer.
        single = np.asarray(adapter.reducer.weights, dtype=np.float64)
        codec = WeightedVectors(np.concatenate((single, single)))
    else:
        raise NotImplementedError(
            "Exx_Mixer_History: carry has no mixer for the reduced fields of this loop; use restart"
        )
    return CarriedAndersonMixer(system.input.mixing, codec)


def recording_hooks(mixer: CarriedAndersonMixer | None, system, hooks: dict, kept: list) -> dict:
    """``hooks`` of the semilocal start with a mixer factory that records its first secant pairs.

    The factory builds the mixer the start would build (``hooks``'
    ``mixer_factory`` or :func:`parsec_python.Mixer.make_mixer`) inside a
    :class:`~parsec_python.Mixer.carried.RecordingMixer`, appended to
    ``kept``.  The spin loops take no mixer factory: their start is not
    recorded (the first inner loop starts without seeds).
    """

    if mixer is None or getattr(system.input.scf, "spin_polarized", False):
        return hooks
    from ..Mixer import make_mixer

    base = hooks.get("mixer_factory") or make_mixer

    def factory(settings):
        recorder = RecordingMixer(base(settings), mixer.codec, mixer.settings.memory)
        kept.append(recorder)
        return recorder

    return {**hooks, "mixer_factory": factory}


def fractional_start(start) -> bool:
    """Whether a semilocal start has a partly filled level (``FRACTIONAL_START``)."""

    channels = getattr(start, "spin", None)
    occupations = (
        [start.occupations] if channels is None else [channel.occupations for channel in channels]
    )
    largest = max(
        (
            float(np.max(np.minimum(values, 1.0 - values), initial=0.0))
            for values in (np.asarray(item, dtype=np.float64) for item in occupations)
        ),
        default=0.0,
    )
    return largest >= FRACTIONAL_START


__all__ = [
    "FIRST_DEXX_RATIO",
    "FIRST_RATIO",
    "FORCING",
    "FRACTIONAL_START",
    "InnerCriterion",
    "LAST_MARGIN",
    "START_CRITERION",
    "fock_loop_mixer",
    "fractional_start",
    "last_loop_iteration",
    "loop_iterations",
    "recording_hooks",
    "selected_residual",
]
