"""Anderson (DIIS) extrapolation of the density matrix of a Fock loop, with a saddle guard.

``Exx_Outer_Mixing: diis`` of :func:`parsec_python.SCF.hybrid.run_hybrid_scf`.
The plain Fock loop is the fixed-point iteration ``X_{n+1} = Y_n = G(X_n)``:
the inner loop ``n`` is solved with the exchange operator of the density
matrix ``X_n`` and returns the density matrix ``Y_n`` of its orbitals.  It
converges linearly, slowly where the hybrid state has a soft mode (a ring of
12 hydrogen atoms: ratio 0.90 per outer iteration) or the exchange is strong
(Hartree-Fock: 0.1-0.3 in dexx).  Anderson's method (Pulay's DIIS with the
outputs, ``beta = 1``) takes instead

``X_{n+1} = sum_k c_k Y_k``,  ``c = argmin || sum_k c_k R_k ||``,  ``sum_k c_k = 1``,

with the residuals ``R_k = Y_k - X_k`` of a window of loops in the Frobenius
norm.  Every ``X_k`` is a combination of earlier outputs, so every residual
is a combination of the stored ``Y`` and ``Tr(R_i R_j)`` follows from
``Tr(Y_a Y_b) = sum_G sum_ij f_i^a f_j^b ((Psi_a^G)^T Psi_b^G)_ij^2`` (one
block ``G`` per symmetry sector: the density matrices are block-diagonal in
the irreducible representations), exactly.

The operator of ``X_{n+1}`` is the exchange operator of a density matrix with
fractional natural occupations (:func:`natural_orbitals`): per channel and
sector, ``X = B C B^T`` with ``B`` the stored occupied orbitals and ``C =
diag(c_k f^k)``; with ``S = B^T B = U L U^T`` (directions below
``OVERLAP_CUTOFF`` of the largest dropped) the matrix ``L^-1/2 U^T S C S U
L^-1/2 = V diag(n) V^T`` gives the natural orbitals ``Phi = B U L^-1/2 V`` and
occupations ``n``.  The ``n_occ`` largest are kept (the others differ from 0
at second order in the changes of the density matrix), clipped to ``[0, 1]``
and scaled to the electron count of the channel; the empty states of the last
loop, orthogonalized against them, complete the ACE (the empty levels).  A
proposal whose kept occupations fall below one half, or whose dropped
occupations exceed ``DROPPED_LIMIT``, is not taken (another occupation
pattern).

When (:meth:`OuterDIIS.judge`, :meth:`OuterDIIS.proposal`): two residuals at least; the last
orbitals with integer occupations in every channel (largest ``min(f, 1 - f)``
below ``FRACTIONAL_GUARD``; a fractional state clears the window: the
symmetric ensemble of a degenerate half-filled pair is a saddle of the
hybrid functional that an extrapolation would be drawn to); ``dexx`` not
risen in the last step (a symmetry breaking under way clears the window
before it); slow convergence (``dexx`` ratio of the last two plain steps
above ``SLOW_RATIO``); and never for the loop that may be the last
(:meth:`parsec_python.SCF.fock_controls.InnerCriterion.predicted_error`
below ``FINAL_MARGIN * Exx_Tolerance``); and a loop solved with an
extrapolated operator never ends the Fock loop
(:func:`parsec_python.SCF.hybrid.run_hybrid_scf`): the final operator is the
plain ACE of the orbitals before, and the stop test is that of the plain loop.  A
DIIS step whose loop raised ``dexx`` or the energy is rejected
(:meth:`OuterDIIS.judge`) and taken back: its output is dropped, the window
keeps the output the step started from, and the next operator is the one of
that density matrix (the plain step the extrapolation replaced; its loop does
not end the Fock loop either, the one after it is plain).  No step is
extrapolated for the last allowed outer iteration (``Exx_Max_Iter``).
Coefficients larger than ``COEFFICIENT_LIMIT`` are not taken.  No orbitals
are copied before that gate is open (:meth:`OuterDIIS.judge`): a loop that
is fast, has a fractional state, a rise of dexx or its end in sight holds
no window, so a run that never extrapolates keeps no extra device memory;
the window starts with the output that opens the gate, and its first
extrapolation follows two loops later (two residuals take three outputs).

The arrays of the orbitals are NumPy (host) or CuPy (the device of each
sector); the algebra on them is the same.  Units Ry.  Nothing here prints or
opens files.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

#: An extrapolated step is not taken when the next dexx is expected below this many Exx_Tolerance
#: (with 3 the ring of 12 hydrogen atoms crept through 13 plain steps at 0.9 from 3e-8 to 1e-8).
FINAL_MARGIN = 1.0

#: dexx ratio of the last two outer steps above which the loop counts as slow.
SLOW_RATIO = 0.05
#: Largest min(f, 1 - f) of an integer occupation.
FRACTIONAL_GUARD = 1.0e-3
#: Outputs kept in the window.
WINDOW = 4
#: Directions of the overlap of the stored orbitals below this share of the largest are dropped.
OVERLAP_CUTOFF = 1.0e-10
#: Natural occupations dropped per channel (beyond the n_occ largest) above which a proposal is not taken.
#: An extrapolation by c along a step d leaves occupations of order (c d)^2 beyond n_occ.
DROPPED_LIMIT = 5.0e-2
#: Largest Anderson coefficient taken: a mode converging linearly with ratio r asks for about
#: 1 / (1 - r) (Aitken), 12 at the 0.92 of the ring of 12 hydrogen atoms, 50 at 0.98.
COEFFICIENT_LIMIT = 100.0
#: Relative regularization of the residual matrix.
REGULARIZATION = 1.0e-12
#: Residuals along one slow mode are nearly parallel: the oldest are dropped while the
#: normalized residual matrix is worse conditioned than this.
CONDITION_LIMIT = 1.0e10


def array_module(array):
    """NumPy or CuPy, whichever holds ``array``."""

    module = type(array).__module__
    if module.startswith("cupy"):
        import cupy

        return cupy
    return np


class _Place:
    """The device of a CuPy array as a context (nothing for NumPy)."""

    def __init__(self, array) -> None:
        self.context = None
        if array_module(array) is not np:
            import cupy

            self.context = cupy.cuda.Device(int(array.device.id))

    def __enter__(self):
        if self.context is not None:
            self.context.__enter__()
        return self

    def __exit__(self, *details):
        if self.context is not None:
            self.context.__exit__(*details)
        return False


def to_host(array) -> np.ndarray:
    xp = array_module(array)
    return np.asarray(array if xp is np else xp.asnumpy(array), dtype=np.float64)


@dataclass
class OccupiedBlock:
    """Copies of the occupied orbitals of one channel in one sector (``rows x n``) and their occupations."""

    vectors: object
    occupations: np.ndarray


@dataclass
class OutputSet:
    """``Y_n``: the occupied orbitals of each channel, by sector key, and the input it was solved with."""

    step: int
    channels: list  # per channel: {key: OccupiedBlock}
    source: dict  # X_n = sum source[k] Y_k (empty for the start)
    error: float | None = None
    energy: float | None = None
    extrapolated: bool = False


def _overlaps(left: OutputSet, right: OutputSet) -> list:
    """Per channel, per common sector key: ``(Psi_left^G)^T Psi_right^G`` on the host."""

    products = []
    for left_channel, right_channel in zip(left.channels, right.channels):
        blocks = {}
        for key, block in left_channel.items():
            other = right_channel.get(key)
            if other is None:
                continue
            with _Place(block.vectors):
                xp = array_module(block.vectors)
                second = other.vectors
                if array_module(second) is not xp or (
                    xp is not np and int(second.device.id) != int(block.vectors.device.id)
                ):
                    second = xp.asarray(to_host(second)) if xp is not np else to_host(second)
                blocks[key] = to_host(block.vectors.T @ second)
        products.append(blocks)
    return products


def _trace(left: OutputSet, right: OutputSet, products) -> float:
    total = 0.0
    for left_channel, right_channel, blocks in zip(left.channels, right.channels, products):
        for key, product in blocks.items():
            weighted = left_channel[key].occupations[:, None] * (product * product) * right_channel[key].occupations[None, :]
            total += float(weighted.sum())
    return total


@dataclass
class OuterDIIS:
    """The window and the decisions of ``Exx_Outer_Mixing: diis`` (see the module)."""

    enabled: bool
    tolerance: float
    window: list = field(default_factory=list)
    products: dict = field(default_factory=dict)
    traces: dict = field(default_factory=dict)
    steps_taken: int = 0
    steps_rejected: int = 0
    last_note: str = ""
    plain_ratio: float | None = None
    #: The step whose output a rejected extrapolation goes back to (read once).
    rollback: int | None = None
    #: The numbers of the last output judged (no orbitals): the plain ratio and the rise test.
    last: OutputSet | None = None

    def _forget(self, keep: list) -> None:
        kept = {entry.step for entry in keep}
        self.window = list(keep)
        self.products = {key: value for key, value in self.products.items() if set(key) <= kept}
        self.traces = {key: value for key, value in self.traces.items() if set(key) <= kept}

    def clear(self) -> None:
        self._forget([])

    def make_room(self) -> None:
        """Drop the oldest output of a full window before the next one is copied."""

        if len(self.window) >= WINDOW:
            self._forget(self.window[-(WINDOW - 1):])

    def judge(self, output: OutputSet, occupations, criteria=None, final_next: bool = False) -> bool:
        """Take the numbers of a finished loop (``error`` dexx, ``energy``) and
        say whether its orbitals join the window.

        Nothing is copied before the loop is judged slow, by the gate of an
        extrapolated step: integer occupations, dexx not risen, the dexx ratio
        of the last two plain steps above ``SLOW_RATIO``, the next loop
        neither expected to be the last (``criteria``) nor the last allowed
        (``final_next``).  A loop that fails it frees the window (a fractional
        state, a rise, a fast loop, the end in sight).  An extrapolated step
        that raised dexx or the energy is rejected: its output is not kept,
        the window is the output it started from, and ``rollback`` names that
        step (the next operator is the one of its density matrix, as if the
        step had not been taken).
        """

        if not self.enabled:
            return False
        self.last_note = ""
        previous, self.last = self.last, replace(output, channels=None)
        fractional = max(
            (float(np.max(np.minimum(values, 1.0 - values), initial=0.0)) for values in occupations),
            default=0.0,
        )
        if fractional > FRACTIONAL_GUARD:
            self._free("fractional occupations: window cleared")
            return False
        if (
            previous is not None
            and not output.extrapolated
            and previous.error is not None
            and output.error is not None
            and previous.error > 0.0
            and set(output.source) == {previous.step}
        ):
            # The speed of the plain fixed point (an extrapolated step would hide it).
            self.plain_ratio = output.error / previous.error
        if previous is not None and previous.error is not None and output.error is not None:
            risen = output.error >= previous.error
            higher = (
                output.energy is not None
                and previous.energy is not None
                and output.energy > previous.energy
            )
            if output.extrapolated and (risen or higher) and self.window:
                origin = self.window[-1]
                self.steps_rejected += 1
                self.last_note = (
                    f"extrapolated step rejected (dexx or energy rose): back to the output of loop {origin.step}"
                )
                self.rollback = origin.step
                self._forget([origin])
                self.last = replace(origin, channels=None)
                return False
            if risen:
                self._free("dexx rose: window cleared")
                return False
        if self.plain_ratio is None or self.plain_ratio <= SLOW_RATIO:
            self._free("fast loop: no window")
            return False
        predicted = None if criteria is None else criteria.predicted_error()
        if final_next or (predicted is not None and predicted < FINAL_MARGIN * self.tolerance):
            self._free("the loop may end: no window")
            return False
        return True

    def _free(self, note: str) -> None:
        """Clear the window (the note only where it held copies)."""

        if self.window:
            self.last_note = note
        self.clear()

    def append(self, output: OutputSet) -> None:
        """Keep a loop's output whose orbitals were copied (after :meth:`judge`)."""

        self.window.append(output)
        for entry in self.window:
            products = _overlaps(entry, output)
            self.products[(entry.step, output.step)] = products
            self.traces[(entry.step, output.step)] = self.traces[(output.step, entry.step)] = _trace(
                entry, output, products
            )
        if len(self.window) > WINDOW:
            self._forget(self.window[-WINDOW:])

    def observe(self, output: OutputSet, occupations, criteria=None, final_next: bool = False) -> bool:
        """:meth:`judge` and, where it lets the output in, :meth:`append` (its orbitals already copied)."""

        keep = self.judge(output, occupations, criteria, final_next)
        if keep:
            self.append(output)
        return keep

    def _residuals(self) -> tuple[list[int], np.ndarray]:
        steps = [entry.step for entry in self.window]
        index = {step: position for position, step in enumerate(steps)}
        rows = []
        for entry in self.window:
            if not entry.source or not set(entry.source) <= set(index):
                continue
            vector = np.zeros(len(steps))
            vector[index[entry.step]] += 1.0
            for step, coefficient in entry.source.items():
                vector[index[step]] -= coefficient
            rows.append(vector)
        return steps, np.array(rows)

    def proposal(self, criteria) -> dict | None:
        """``{step: c}`` of the next input ``X = sum c Y``, or ``None`` for the plain step."""

        if not self.enabled or not self.window:
            return None
        if len(self.window) < 3:
            # The window started at the gate (judge): two residuals take three outputs.
            self.last_note = f"slow loop: {len(self.window)} of the 3 outputs of an extrapolation kept"
            return None
        if self.plain_ratio is None or self.plain_ratio <= SLOW_RATIO:
            self._free("fast loop: no window")
            return None
        predicted = criteria.predicted_error()
        if predicted is not None and predicted < FINAL_MARGIN * criteria.tolerance:
            self.last_note = "the next loop may be the last: plain step"
            return None
        steps, residuals = self._residuals()
        if residuals.shape[0] < 2:
            return None
        gram = np.array([[self.traces[(a, b)] for b in steps] for a in steps])
        # The residual rows belong to the outputs of the window whose inputs it holds.
        holders = [entry.step for entry in self.window if entry.source and set(entry.source) <= set(steps)]
        while True:
            matrix = residuals @ gram @ residuals.T
            matrix = 0.5 * (matrix + matrix.T)
            scale = float(np.max(np.abs(np.diag(matrix)), initial=0.0))
            if scale <= 0.0:
                return None
            matrix = matrix / scale + REGULARIZATION * np.eye(matrix.shape[0])
            values = np.linalg.eigvalsh(matrix)
            if residuals.shape[0] <= 2 or values[0] > values[-1] / CONDITION_LIMIT:
                break
            residuals, holders = residuals[1:], holders[1:]
        try:
            weights = np.linalg.solve(matrix, np.ones(matrix.shape[0]))
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(weights)) or abs(float(weights.sum())) < 1.0e-300:
            return None
        weights = weights / float(weights.sum())
        coefficients = {step: float(weight) for step, weight in zip(holders, weights)}
        if max(abs(value) for value in coefficients.values()) > COEFFICIENT_LIMIT:
            self.last_note = "coefficients too large: plain step"
            return None
        return coefficients

    def entry(self, step: int) -> OutputSet:
        for entry in self.window:
            if entry.step == step:
                return entry
        raise KeyError(step)

    def natural_orbitals(self, coefficients: dict) -> list | None:
        """Per channel ``{key: (Phi, n)}`` of ``X = sum c_k Y_k``, or ``None`` (not taken).

        The occupied states of each sector and the electrons of each channel
        are those of the last output of the window.
        """

        latest = self.window[-1]
        electrons = [
            sum(float(block.occupations.sum()) for block in channel.values()) for channel in latest.channels
        ]
        occupied_counts = [
            {key: int(block.occupations.size) for key, block in channel.items()} for channel in latest.channels
        ]
        steps = [step for step, value in coefficients.items() if value != 0.0]
        entries = [self.entry(step) for step in steps]
        channels = []
        for channel, total in enumerate(electrons):
            keys = sorted(
                {key for entry in entries for key in entry.channels[channel]}, key=lambda item: (item is None, item)
            )
            blocks = {}
            dropped = 0.0
            kept_sum = 0.0
            for key in keys:
                members = [
                    (entry, coefficients[entry.step])
                    for entry in entries
                    if key in entry.channels[channel]
                ]
                sizes = [entry.channels[channel][key].occupations.size for entry, _ in members]
                offsets = np.concatenate([[0], np.cumsum(sizes)])
                count = int(offsets[-1])
                overlap = np.zeros((count, count))
                for i, (left, _) in enumerate(members):
                    for j, (right, _) in enumerate(members):
                        pair = self.products.get((left.step, right.step))
                        if pair is not None:
                            block = pair[channel][key]
                        else:
                            block = self.products[(right.step, left.step)][channel][key].T
                        overlap[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]] = block
                overlap = 0.5 * (overlap + overlap.T)
                weights = np.concatenate(
                    [value * entry.channels[channel][key].occupations for entry, value in members]
                )
                values, vectors = np.linalg.eigh(overlap)
                keep = values > OVERLAP_CUTOFF * max(float(values.max(initial=0.0)), 0.0)
                basis = vectors[:, keep] / np.sqrt(values[keep])
                projected = basis.T @ overlap @ (weights[:, None] * (overlap @ basis))
                projected = 0.5 * (projected + projected.T)
                occupations, rotation = np.linalg.eigh(projected)
                order = np.argsort(occupations)[::-1]
                wanted = int(occupied_counts[channel].get(key, 0))
                chosen = order[:wanted]
                dropped += float(np.sum(np.abs(occupations[order[wanted:]])))
                natural = occupations[chosen]
                if natural.size and float(natural.min()) < 0.5:
                    self.last_note = "another occupation pattern: plain step"
                    return None
                natural = np.clip(natural, 0.0, 1.0)
                kept_sum += float(natural.sum())
                combination = basis @ rotation[:, chosen]
                phi = None
                for i, (entry, _) in enumerate(members):
                    part = combination[offsets[i] : offsets[i + 1]]
                    stored = entry.channels[channel][key].vectors
                    with _Place(stored):
                        xp = array_module(stored)
                        term = stored @ xp.asarray(part)
                        phi = term if phi is None else phi + _same_place(term, phi)
                blocks[key] = [phi, natural]
            if dropped > DROPPED_LIMIT:
                self.last_note = f"natural occupations beyond the occupied states ({dropped:.1E}): plain step"
                return None
            if kept_sum > 0.0:
                for key in blocks:
                    blocks[key][1] = np.minimum(blocks[key][1] * (float(total) / kept_sum), 1.0)
            channels.append({key: (phi, natural) for key, (phi, natural) in blocks.items()})
        return channels


def _same_place(term, reference):
    xp = array_module(reference)
    if xp is np:
        return to_host(term)
    if array_module(term) is np or int(term.device.id) != int(reference.device.id):
        return xp.asarray(to_host(term))
    return term


def orthonormal_complement(phi, empty):
    """``empty`` orthogonalized against the orthonormal columns ``phi`` and among themselves (Loewdin)."""

    with _Place(phi):
        xp = array_module(phi)
        empty = _same_place(empty, phi)
        rest = empty - phi @ (phi.T @ empty)
        gram = to_host(rest.T @ rest)
        values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
        keep = values > 1.0e-8 * max(float(values.max(initial=0.0)), 1.0e-300)
        transform = vectors[:, keep] / np.sqrt(values[keep])
        return rest @ xp.asarray(transform)


__all__ = [
    "COEFFICIENT_LIMIT",
    "CONDITION_LIMIT",
    "DROPPED_LIMIT",
    "FRACTIONAL_GUARD",
    "OccupiedBlock",
    "OuterDIIS",
    "OutputSet",
    "SLOW_RATIO",
    "WINDOW",
    "array_module",
    "orthonormal_complement",
    "to_host",
]
