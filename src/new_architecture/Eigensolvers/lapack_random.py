"""Pure-Python LAPACK random stream used by PARSEC ``random_array``.

PARSEC's real ``random_array`` routine in ``src/dgks.f90z`` calls
``DLARNV(IDIST=2, ...)`` once per vector column and keeps the four-integer
LAPACK seed between calls.  LAPACK's underlying ``DLARUV`` generator is a
48-bit multiplicative congruential generator.  Representing that 48-bit
integer directly makes the routine considerably easier to read in Python
than the portable four-base-4096 arithmetic used by the Fortran 77 source.

For integer state ``s_k`` and multiplier ``a``, the stream is

``s_(k+1) = a*s_k mod 2^48``

``u_(k+1) = s_(k+1)/2^48``.

``DLARNV(IDIST=2)`` maps ``u`` from ``(0,1)`` to ``2*u-1`` in ``(-1,1)``.
``run_chebff`` uses this stream for its initial basis and any dependent-vector
replacements in the following orthonormalization.  Short-Lanczos starts use a
separate NumPy stream.  The Python stream is restarted on an explicit CHEBFF
restart; PARSEC's saved DLARNV seed is process-persistent, so this module is
sequence-exact within one CHEBFF run rather than across every possible restart.

This is native Python integer arithmetic.  It does not call or bind to
LAPACK/Fortran at run time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_BASE = 4096
_MODULUS = 1 << 48
_MULTIPLIER = 33_952_834_046_453
PARSEC_RANDOM_ARRAY_SEED = (1, 1000, 2000, 4095)


def _validate_seed(seed: Iterable[int]) -> tuple[int, int, int, int]:
    values = tuple(int(value) for value in seed)
    if len(values) != 4:
        raise ValueError("a LAPACK random seed has exactly four integers")
    if any(value < 0 or value >= _BASE for value in values):
        raise ValueError("LAPACK seed entries must be between 0 and 4095")
    if values[-1] % 2 != 1:
        raise ValueError("the fourth LAPACK seed entry must be odd")
    return values


def _seed_to_integer(seed: tuple[int, int, int, int]) -> int:
    state = 0
    for digit in seed:
        state = _BASE * state + digit
    return state


def _integer_to_seed(state: int) -> tuple[int, int, int, int]:
    digits = [0, 0, 0, 0]
    for index in range(3, -1, -1):
        state, digits[index] = divmod(state, _BASE)
    return tuple(digits)  # type: ignore[return-value]


@dataclass
class LapackRandom:
    """Stateful translation of the ``DLARUV``/``DLARNV`` uniform path.

    The four base-4096 seed digits are LAPACK's external representation of one
    48-bit state.  Every draw updates ``seed``, so reusing this object across
    basis creation and recovery vectors preserves stream order.
    """

    seed: tuple[int, int, int, int] = PARSEC_RANDOM_ARRAY_SEED

    def __post_init__(self) -> None:
        self.seed = _validate_seed(self.seed)

    def uniform_0_1(self, count: int) -> np.ndarray:
        """Return ``count`` values from LAPACK's uniform ``(0, 1)`` stream."""

        count = int(count)
        if count < 0:
            raise ValueError("count cannot be negative")
        # Convert the portable four-digit seed once, advance the 48-bit state
        # for each draw, then expose the final state again as four digits.
        state = _seed_to_integer(self.seed)
        values = np.empty(count, dtype=np.float64)
        scale = 1.0 / _MODULUS
        for index in range(count):
            state = (_MULTIPLIER * state) % _MODULUS
            values[index] = state * scale
        self.seed = _integer_to_seed(state)
        return values

    def uniform_minus_1_1(
        self,
        shape: int | tuple[int, ...],
        *,
        column_major: bool = False,
    ) -> np.ndarray:
        """Return PARSEC ``DLARNV(IDIST=2)`` values with the requested shape.

        ``random_array`` fills one complete Fortran column before advancing
        to the next.  Pass ``column_major=True`` for a matrix destined to be
        used as a column-vector basis.
        """

        if isinstance(shape, int):
            normalized_shape = (shape,)
        else:
            normalized_shape = tuple(int(value) for value in shape)
        if any(value < 0 for value in normalized_shape):
            raise ValueError("shape entries cannot be negative")
        count = int(np.prod(normalized_shape, dtype=np.int64))
        # LAPACK IDIST=2 is exactly the affine transformation x=2*u-1.
        values = 2.0 * self.uniform_0_1(count) - 1.0
        order = "F" if column_major else "C"
        return values.reshape(normalized_shape, order=order)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray | float:
        """NumPy-compatible uniform draw backed by the LAPACK stream."""

        if high <= low:
            raise ValueError("high must be greater than low")
        if size is None:
            return float(low + (high - low) * self.uniform_0_1(1)[0])
        if isinstance(size, int):
            shape = (size,)
        else:
            shape = tuple(int(value) for value in size)
        count = int(np.prod(shape, dtype=np.int64))
        values = low + (high - low) * self.uniform_0_1(count)
        return values.reshape(shape)


__all__ = ["LapackRandom", "PARSEC_RANDOM_ARRAY_SEED"]
