"""PARSEC's generalized Broyden potential mixer (``genbro.f90``).

This is the algorithm of M. Jain and L. Kronik that ``genbro.f90`` implements,
not D. D. Johnson's modified Broyden method.  With ``f_i = x_out,i - x_in,i``,
``df_i = f_i - f_(i-1)``, ``dx_i = x_in,i - x_in,(i-1)`` and the scalar first
inverse Jacobian ``B_1 = -alpha`` (``mixer%resid1 = -Mixing_Param``):

* ``B_n = B_1 + sum_(i=2..n) u_i v_i^T``,
* ``u_i = dx_i - B_1 df_i - sum_(k=2..i-1) u_k (v_k . df_i)``,
* ``T_ij = df_i . df_j`` over the last ``p = Memory_Param`` differences,
* ``v_i = sum_k [T^-1]_(p,k) df_(i-p+k)``, so ``v_i . df_i = 1`` and ``v_i``
  is orthogonal to the other differences of the window,
* ``x_new = x_in - B_n f_n``.

Every pair ``(u_i, v_i)`` since the last restart enters ``B_n``; only the
window of differences in ``T`` is limited to ``p``.  The history restarts
every ``Restart_Mixing`` iterations (``bitr = mod(iter-1, restart) + 1``), and
``p = 0`` is linear mixing ``(1-alpha) x_in + alpha x_out``.

Deviation from the Fortran file layout: ``genbro.f90`` keeps the pairs in
files ``bro.<block>.<rank>`` of ``block - 1`` slots (``mixer%block`` is 20, or
``Memory_Param + 1`` when larger).  Its read and write counts disagree once a
restart cycle passes ``block`` iterations: the pair of step
``bitr = 1 + k (block - 1)`` (k >= 1) is written to a new file that the next
step does not read, so for ``Restart_Mixing > block`` PARSEC drops those pairs
from every later step of the cycle.  The default ``Restart_Mixing = 20`` never
reaches that step.  This port keeps every pair in memory, the algorithm of the
equations above, and so agrees with PARSEC whenever ``Restart_Mixing`` is at
most ``block``; :mod:`parsec_python.Input.parsec_input` warns otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import lapack

from ..models import MixingSettings


def _validated_weights(
    weights: np.ndarray | None, size: int | None = None
) -> np.ndarray | None:
    """Return float64 inner-product weights, or ``None`` for plain dots."""

    if weights is None:
        return None
    array = np.ascontiguousarray(weights, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("mixer weights must be one positive finite value per entry")
    if size is not None and array.size != size:
        raise ValueError("mixer weights do not match the mixed vector")
    return array


@dataclass
class GeneralizedBroydenMixer:
    """Potential mixer matching ``genbro.f90`` (``Mixing_Method: broyden``).

    ``mix(x_in, x_out, iteration=k)`` returns the next input potential.  Both
    vectors are one-dimensional: the local potential on the grid, or for a
    spin-polarized calculation ``[V_up; V_down]`` (``spinmix.f90``).

    ``weights`` gives every inner product the form ``sum_i w_i a_i b_i``.
    ``genbro.f90`` multiplies its wedge dot products by the group order
    ``nrep``, a uniform weight that cancels in every formula; the full grid
    therefore needs none, and a symmetry wedge whose orbits differ in size
    takes the orbit multiplicities, which reproduce the full-grid products.

    ``singular_warnings`` counts the inversions of ``T`` in which LAPACK met
    an exactly zero pivot, where ``genbro.f90`` writes ``Warning: T matrix in
    mixer is Singular`` and goes on with what ``dgetri`` left.
    """

    settings: MixingSettings = field(
        default_factory=lambda: MixingSettings(method="broyden")
    )
    weights: np.ndarray | None = None
    singular_warnings: int = field(default=0, init=False)
    # genbro.f90's u(:,2:) and v(:,2:) since the last restart, its window
    # df(:,1:p) of residual differences and the matrix t(1:p,1:p) of their
    # inner products; old1 and old2 are the previous input and output.
    _u: list[np.ndarray] = field(default_factory=list, init=False)
    _v: list[np.ndarray] = field(default_factory=list, init=False)
    _differences: list[np.ndarray] = field(default_factory=list, init=False)
    _gram: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0)), init=False
    )
    _previous_input: np.ndarray | None = field(default=None, init=False)
    _previous_output: np.ndarray | None = field(default=None, init=False)
    _calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.weights = _validated_weights(self.weights)

    def reset(self) -> None:
        self._clear_history()
        self._previous_input = None
        self._previous_output = None
        self._calls = 0
        self.singular_warnings = 0

    def _clear_history(self) -> None:
        self._u.clear()
        self._v.clear()
        self._differences.clear()
        self._gram = np.zeros((0, 0))

    def _dot(self, left: np.ndarray, right: np.ndarray) -> float:
        if self.weights is None:
            return float(np.dot(left, right))
        return float(np.dot(left, self.weights * right))

    def _inverse(self, matrix: np.ndarray) -> np.ndarray:
        """``dgetrf`` + ``dgetri`` as ``genbro.f90:251-266`` calls them."""

        factors, pivots, info = lapack.dgetrf(matrix)
        if info < 0:
            raise RuntimeError(f"dgetrf failed in the Broyden mixer (info={info})")
        if info > 0:
            self.singular_warnings += 1
        inverse, info = lapack.dgetri(factors, pivots)
        if info < 0:
            raise RuntimeError(f"dgetri failed in the Broyden mixer (info={info})")
        if info > 0:
            self.singular_warnings += 1
        return inverse

    def mix(
        self,
        input_potential: np.ndarray,
        output_potential: np.ndarray,
        *,
        iteration: int | None = None,
    ) -> np.ndarray:
        x = np.asarray(input_potential, dtype=float)
        y = np.asarray(output_potential, dtype=float)
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError(
                "input and output potentials must be identical one-dimensional vectors"
            )
        if self.weights is not None and self.weights.size != x.size:
            raise ValueError("mixer weights do not match the mixed vector")
        if iteration is None:
            iteration = self._calls + 1
        if iteration < 1:
            raise ValueError("SCF iteration numbers start at one")
        self._calls = iteration
        alpha = self.settings.parameter
        memory = self.settings.memory
        # B_1 = resid1 = -Mixing_Param (genbro.f90:160).
        jacobian = -alpha

        if memory == 0:
            # genbro.f90:144-147: linear mixing.
            return (1 - alpha) * x + alpha * y

        step = (iteration - 1) % self.settings.restart + 1
        if step == 1:
            # genbro.f90:154-174: start of the history, x + alpha f.
            self._clear_history()
            self._previous_input = x.copy()
            self._previous_output = y.copy()
            return x - jacobian * (y - x)
        if self._previous_input is None or self._previous_output is None:
            raise ValueError(
                "the Broyden history starts at an iteration k with "
                "mod(k-1, Restart_Mixing) = 0"
            )

        # genbro.f90:208-221 / 335-340 and the u loop of 409-420: the new
        # difference df = f - f_old and u = dx - B_1 df - sum_j u_j (v_j.df).
        input_step = x - self._previous_input
        difference = y - self._previous_output - input_step
        u = input_step - jacobian * difference
        for previous_u, previous_v in zip(self._u, self._v):
            u -= self._dot(previous_v, difference) * previous_u

        # genbro.f90:226-231 / 352-366: the window of differences and their
        # matrix, shifted by one once it holds p of them.
        products = np.array(
            [self._dot(difference, kept) for kept in self._differences]
            + [self._dot(difference, difference)]
        )
        if len(self._differences) == memory:
            self._differences.pop(0)
            products = products[1:]
            kept_gram = self._gram[1:, 1:]
        else:
            kept_gram = self._gram
        self._differences.append(difference)
        size = len(self._differences)
        gram = np.empty((size, size))
        gram[:-1, :-1] = kept_gram
        gram[-1, :] = products
        gram[:, -1] = products
        self._gram = gram

        # genbro.f90:269-273 / 403-407: v = sum_k [T^-1]_(last,k) df_k.
        inverse = self._inverse(gram)
        v = np.zeros_like(x)
        for coefficient, kept in zip(inverse[-1], self._differences):
            v += coefficient * kept
        self._u.append(u)
        self._v.append(v)
        self._previous_input = x.copy()
        self._previous_output = y.copy()

        # genbro.f90:283-300 / 346-510: x_new = x - B_n f with
        # B_n f = B_1 f + sum_j u_j (v_j.x_out - v_j.x_in).
        update = jacobian * (y - x)
        for kept_u, kept_v in zip(self._u, self._v):
            update += (self._dot(kept_v, y) - self._dot(kept_v, x)) * kept_u
        return x - update


__all__ = ["GeneralizedBroydenMixer"]
