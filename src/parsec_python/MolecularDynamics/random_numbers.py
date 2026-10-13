"""Random numbers of PARSEC's molecular dynamics.

Translation of the three functions at the end of ``moldyn.f90``:

``gdev``
    one Gaussian deviate from two uniform numbers (polar Box-Muller);
``get_rand_test``
    PARSEC's own uniform generator, selected by the input ``test_rng``;
``getrandom``
    the default: uniform numbers of the compiler's ``random_number``, served
    from a buffer of 10000.

``get_rand_test`` is a complete algorithm, so :class:`ParsecTestRandom`
returns exactly the numbers of the Fortran program.  ``getrandom`` depends on
the Fortran compiler and, without ``rng_seed``, on the system clock, so its
numbers are not a property of PARSEC.  :class:`BufferedRandom` keeps the
structure of ``getrandom`` and takes the numbers from an explicit NumPy
generator instead; this is the default.  :class:`GfortranRandom` returns the
numbers of one particular compiler, gfortran, for a run with ``rng_seed``.

The two buffered sources have a ``position``, the Fortran
``mol_dynamic%iset``: the place of the next number in the buffer, counted
from one, and zero when the buffer has to be filled first.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .settings import RANDOM_BUFFER_SIZE


class UniformSource(Protocol):
    """What :func:`gaussian_deviate` needs from a generator."""

    #: ``gdev`` discards a point within 1e-7 of the origin only when it uses
    #: ``get_rand_test``.
    rejects_small_radius: bool

    def uniform(self) -> float:
        """Return the next uniform number of [0, 1)."""
        ...


class ParsecTestRandom:
    """``get_rand_test``: the minimal standard generator of Park and Miller.

    ``seed(n+1) = 16807*seed(n) mod (2**31 - 1)``, evaluated with Schrage's
    factorisation so that it fits a 32-bit integer, and returned as
    ``seed/(2**31 - 1)``.  The state is ``mol_dynamic%rng_seed`` itself: the
    input ``rng_seed`` is the first state, and a state of zero is replaced by
    123459876.  The Fortran comment says it is "for testing purposes only".

    ``seed`` is the current state and can be read after every call.
    """

    rejects_small_radius = True

    _IA = 16807
    _IM = 2147483647
    _IQ = 127773
    _IR = 2836
    _MASK = 123459876

    def __init__(self, seed: int = 0) -> None:
        seed = int(seed)
        if not -(2**31) <= seed < 2**31:
            raise ValueError("rng_seed must fit a 32-bit integer")
        self.seed = seed

    def uniform(self) -> float:
        iseed = self.seed
        # ``if (iseed > im) iseed = mod(iseed,im)`` can never hold for a
        # 32-bit integer: im is the largest one.
        if iseed == 0:
            iseed = self._MASK
        # Fortran integer division truncates toward zero.
        kk = abs(iseed) // self._IQ
        if iseed < 0:
            kk = -kk
        iseed = self._IA * (iseed - kk * self._IQ) - self._IR * kk
        if iseed < 0:
            iseed += self._IM
        self.seed = iseed
        return iseed / self._IM


class BufferedRandom:
    """``getrandom`` with a NumPy generator in place of ``random_number``.

    PARSEC fills ``mol_dynamic%vrandom`` with ``nrandom = 10000`` uniform
    numbers and hands them out one by one; when the buffer is used up it
    seeds the compiler generator again and refills it.  The seed it puts is

    * ``rng_seed`` itself when the input ``rng_seed`` is positive.  Every
      refill then starts from the same seed, so the same 10000 numbers come
      back: with ``rng_seed`` the Fortran random sequence has a period of
      10000.  This is reproduced with ``numpy.random.default_rng(rng_seed)``
      (the numbers themselves are NumPy's, not the Fortran compiler's);
    * otherwise the current seed plus the system clock.  Here the caller's
      ``generator`` simply continues.

    ``generator`` is a ``numpy.random.Generator`` or a seed for one; ``None``
    takes fresh entropy from the operating system, which is the counterpart
    of PARSEC's clock.
    """

    rejects_small_radius = False

    def __init__(
        self,
        generator: np.random.Generator | int | None = None,
        *,
        rng_seed: int = 0,
        size: int = RANDOM_BUFFER_SIZE,
    ) -> None:
        if size < 1:
            raise ValueError("the random buffer needs at least one number")
        self.rng_seed = int(rng_seed)
        self.size = int(size)
        if self.rng_seed > 0:
            self._generator = None
        elif isinstance(generator, np.random.Generator):
            self._generator = generator
        else:
            self._generator = np.random.default_rng(generator)
        self._buffer = np.empty(0)
        # mol_dynamic%iset: 0 means "refill before the next number"
        self._iset = 0

    @property
    def position(self) -> int:
        """``mol_dynamic%iset``."""
        return self._iset

    def uniform(self) -> float:
        if self._iset == 0:
            if self.rng_seed > 0:
                self._generator = np.random.default_rng(self.rng_seed)
            self._buffer = self._generator.random(self.size)
            self._iset = 1
        value = float(self._buffer[self._iset - 1])
        self._iset += 1
        if self._iset > self.size:
            self._iset = 0
        return value


class GfortranRandom:
    """``getrandom`` as a PARSEC compiled by gfortran runs it with ``rng_seed``.

    ``random_number`` of gfortran 8 and later is xoshiro256**.  PARSEC sets
    every element of the seed array to ``rng_seed``; the library XORs the
    four 64-bit words of that array with fixed keys to get the state, and a
    double precision number is the upper 53 bits of an output word.  With
    this the numbers of such a PARSEC run are reproduced exactly, so a
    Langevin trajectory or the initial velocities of a run without
    ``test_rng`` can be repeated in Python.  It is of no use for a PARSEC
    built by another compiler, and there is nothing to reproduce without
    ``rng_seed``, when PARSEC seeds from the system clock.

    As in :class:`BufferedRandom`, the 10000 numbers of the buffer repeat,
    because PARSEC puts the same seed before every refill.
    """

    rejects_small_radius = False

    _MASK = (1 << 64) - 1
    # xor_keys of libgfortran/intrinsics/random.c
    _KEYS = (
        0xBD0C5B6E50C2DF49,
        0xD46061CD46E1DF38,
        0xBB4F4D4ED6103544,
        0x114A583D0756AD39,
    )

    def __init__(self, rng_seed: int, *, size: int = RANDOM_BUFFER_SIZE) -> None:
        rng_seed = int(rng_seed)
        if not 0 < rng_seed < 2**31:
            raise ValueError(
                "rng_seed must be a positive 32-bit integer: without it PARSEC "
                "seeds the compiler generator from the system clock"
            )
        if size < 1:
            raise ValueError("the random buffer needs at least one number")
        self.rng_seed = rng_seed
        self.size = int(size)
        self._buffer: list[float] = []
        self._iset = 0

    @property
    def position(self) -> int:
        """``mol_dynamic%iset``."""
        return self._iset

    def _refill(self) -> None:
        mask = self._MASK
        # two equal 32-bit integers form each 64-bit word of the seed
        word = (self.rng_seed << 32) | self.rng_seed
        s0, s1, s2, s3 = (word ^ key for key in self._KEYS)
        buffer = []
        for _ in range(self.size):
            product = (s1 * 5) & mask
            result = ((((product << 7) | (product >> 57)) & mask) * 9) & mask
            shifted = (s1 << 17) & mask
            s2 ^= s0
            s3 ^= s1
            s1 ^= s2
            s0 ^= s3
            s2 ^= shifted
            s3 = ((s3 << 45) | (s3 >> 19)) & mask
            buffer.append((result >> 11) * 2.0**-53)
        self._buffer = buffer

    def uniform(self) -> float:
        if self._iset == 0:
            self._refill()
            self._iset = 1
        value = self._buffer[self._iset - 1]
        self._iset += 1
        if self._iset > self.size:
            self._iset = 0
        return value


def gaussian_deviate(source: UniformSource) -> float:
    """``gdev``: a Gaussian deviate of zero mean and unit variance.

    Two uniform numbers give a point ``(v1, v2)`` of the square
    ``[-1, 1)**2``; points outside the unit circle are drawn again.  With
    ``r = v1**2 + v2**2`` the Box-Muller pair is ``v1*fac`` and ``v2*fac``,
    ``fac = sqrt(-2 ln(r)/r)``.  PARSEC returns the second one and throws the
    first away, so every deviate costs at least two uniform numbers.
    """
    while True:
        v1 = 2.0 * source.uniform() - 1.0
        v2 = 2.0 * source.uniform() - 1.0
        r = v1 * v1 + v2 * v2
        if source.rejects_small_radius and r < 1.0e-7:
            continue
        if r < 1.0:
            break
    if r == 0.0:
        # Only without the small-radius test: Fortran evaluates 0*infinity.
        return math.nan
    fac = math.sqrt(-2.0 * math.log(r) / r)
    return v2 * fac
