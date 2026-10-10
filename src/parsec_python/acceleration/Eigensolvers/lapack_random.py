"""Bit-exact host and device generation of PARSEC's DLARNV stream.

The readable implementation advances the 48-bit linear congruential
generator one value at a time.  That is ideal as an explanation and very
costly for an ``N x states`` GPU trial basis.  If ``L`` consecutive states
are initialized once, every lane can jump to its next value independently:

``s[k + L] = a**L * s[k] mod 2**48``.

NumPy then advances 2,048 lanes per operation.  The values and final LAPACK
seed remain bit-for-bit identical. The optional device implementation uses
65,536 independent lanes with the same modular skip-ahead recurrence.
"""

from __future__ import annotations

import numpy as np
from math import prod
from threading import Lock

from parsec_python.Eigensolvers.lapack_random import (
    LapackRandom as _ReadableLapackRandom,
    PARSEC_RANDOM_ARRAY_SEED,
)


_BASE = 4096
_MODULUS = 1 << 48
_MASK = np.uint64(_MODULUS - 1)
_MULTIPLIER = 33_952_834_046_453
# A lane is initialized by a dependent scalar LCG step, whereas all later
# jumps are vectorized.  Sweeps from 512 through 32,768 on the 65k-row sector
# sizes used by the solver put the crossover near 2,048: more lanes spend
# unnecessary time in Python, fewer lanes issue too many small NumPy passes.
# Lane count does not enter the random-number definition, so this is a pure
# execution-tiling choice and the stream/final seed remain bit exact.
_LANES = 2_048

_DEVICE_KERNELS = {}
_DEVICE_KERNEL_LOCK = Lock()
_DEVICE_SOURCE = r"""
extern "C" __global__ void lapack_uniform_minus_1_1(
    const unsigned long long seed, const unsigned long long multiplier,
    const unsigned long long jump, const long long count,
    const long long lanes, double* output) {
    const long long lane = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    const unsigned long long mask = (1ULL << 48) - 1;
    unsigned long long factor = 1, base = multiplier;
    unsigned long long exponent = (unsigned long long)lane + 1;
    while (exponent) {
        if (exponent & 1) factor = (factor * base) & mask;
        base = (base * base) & mask;
        exponent >>= 1;
    }
    unsigned long long state = (seed * factor) & mask;
    for (long long index = lane; index < count; index += lanes) {
        output[index] = 2.0 * ((double)state * (1.0 / 281474976710656.0)) - 1.0;
        state = (state * jump) & mask;
    }
}
"""


def _seed_to_integer(seed: tuple[int, int, int, int]) -> int:
    state = 0
    for digit in seed:
        state = _BASE * state + int(digit)
    return state


def _integer_to_seed(state: int) -> tuple[int, int, int, int]:
    digits = [0, 0, 0, 0]
    for index in range(3, -1, -1):
        state, digits[index] = divmod(state, _BASE)
    return tuple(digits)  # type: ignore[return-value]


class LapackRandom(_ReadableLapackRandom):
    """Drop-in accelerated version of the readable stateful generator."""

    def device_uniform_minus_1_1(self, shape, *, column_major=False, out=None):
        """Generate the identical DLARNV stream directly on the current GPU.

        Unsigned multiplication with a 48-bit mask gives exact modular skip
        ahead. Integer-to-double conversion and power-of-two scaling are exact.
        The host seed advances by the same count, including subsequent CPU
        calls used by replacement-capable orthogonalization.  ``out`` may
        supply a contiguous float64 device array of ``shape`` in the
        requested order to fill instead of allocating one.
        """
        from ..backends.cupy import require_cupy
        from ..backends.cupy_compile import compile_cupy_raw

        cp, _ = require_cupy()
        dimensions = (
            (int(shape),)
            if isinstance(shape, (int, np.integer))
            else tuple(map(int, shape))
        )
        if any(n < 0 for n in dimensions):
            raise ValueError("shape dimensions cannot be negative")
        count = prod(dimensions)
        if out is None:
            output = cp.empty(
                dimensions, dtype=cp.float64, order="F" if column_major else "C"
            )
        else:
            contiguous = out.flags.f_contiguous if column_major else out.flags.c_contiguous
            if tuple(out.shape) != dimensions or out.dtype != cp.dtype(cp.float64) or not contiguous:
                raise ValueError("out must be a contiguous float64 array of the requested shape and order")
            output = out
        if not count:
            return output
        device = int(cp.cuda.Device().id)
        with _DEVICE_KERNEL_LOCK:
            kernel = _DEVICE_KERNELS.get(device)
            if kernel is None:
                kernel = cp.RawKernel(
                    _DEVICE_SOURCE,
                    "lapack_uniform_minus_1_1",
                    options=("--std=c++11", "--fmad=false"),
                )
                compile_cupy_raw(kernel)
                _DEVICE_KERNELS[device] = kernel
        lanes = min(count, 65536)
        state = _seed_to_integer(self.seed)
        jump = pow(_MULTIPLIER, lanes, _MODULUS)
        kernel(
            ((lanes + 255) // 256,),
            (256,),
            (
                np.uint64(state),
                np.uint64(_MULTIPLIER),
                np.uint64(jump),
                np.int64(count),
                np.int64(lanes),
                output,
            ),
        )
        self.seed = _integer_to_seed(
            (state * pow(_MULTIPLIER, count, _MODULUS)) % _MODULUS
        )
        return output

    def uniform_0_1(self, count: int) -> np.ndarray:
        """Return the exact LAPACK stream using vectorized skip-ahead lanes."""

        count = int(count)
        if count < 0:
            raise ValueError("count cannot be negative")
        if count == 0:
            return np.empty(0, dtype=np.float64)

        state = _seed_to_integer(self.seed)
        lane_count = min(count, _LANES)
        states = np.empty(lane_count, dtype=np.uint64)
        # Only this short prefix is dependent scalar work.  Subsequent blocks
        # are independent applications of the exact L-step transition.
        for index in range(lane_count):
            state = (_MULTIPLIER * state) % _MODULUS
            states[index] = state

        values = np.empty(count, dtype=np.float64)
        scale = 1.0 / _MODULUS
        values[:lane_count] = states * scale
        offset = lane_count
        if offset < count:
            jump = np.uint64(pow(_MULTIPLIER, lane_count, _MODULUS))
            while offset < count:
                # Unsigned overflow discards high bits; masking then gives
                # multiplication modulo 2**48 exactly.
                states = np.bitwise_and(states * jump, _MASK)
                take = min(lane_count, count - offset)
                values[offset : offset + take] = states[:take] * scale
                state = int(states[take - 1])
                offset += take

        self.seed = _integer_to_seed(state)
        return values


__all__ = ["LapackRandom", "PARSEC_RANDOM_ARRAY_SEED"]
