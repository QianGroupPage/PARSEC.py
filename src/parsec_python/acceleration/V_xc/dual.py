"""Forward derivatives of a formula on arrays of NumPy or CuPy: ``Dual(value, slope)``.

The partial derivatives of the spin-polarized PBE of the reference drivers
(:func:`parsec_python.V_xc.pbe.pbe_spin_polarized_partials`) are taken by
complex-step differentiation: the formula on ``x + i h`` with ``h = 1e-30``,
the derivative ``Im f / h``.  With a step that small the imaginary parts
follow the chain rule exactly (no second-order term survives), so the same
formula on these pairs gives the same partials to round-off, on arrays of the
device, where complex ``log1p`` and ``expm1`` are not at hand.

Only what that formula uses is defined: sums, differences, products and
quotients with numbers and other pairs, negation, a real power, and the
``log1p`` and ``expm1`` of :class:`DualMath`.
"""

from __future__ import annotations


class Dual:
    """A value and its derivative along one direction (arrays of one module, or numbers)."""

    __slots__ = ("value", "slope")
    __array_priority__ = 1000.0  # NumPy scalars on the left defer to the reflected operations

    def __init__(self, value, slope) -> None:
        self.value = value
        self.slope = slope

    # -- arithmetic --------------------------------------------------------
    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.value + other.value, self.slope + other.slope)
        return Dual(self.value + other, self.slope)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.value - other.value, self.slope - other.slope)
        return Dual(self.value - other, self.slope)

    def __rsub__(self, other):
        return Dual(other - self.value, -self.slope)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.value * other.value, self.value * other.slope + self.slope * other.value)
        return Dual(self.value * other, self.slope * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Dual):
            value = self.value / other.value
            return Dual(value, (self.slope - value * other.slope) / other.value)
        return Dual(self.value / other, self.slope / other)

    def __rtruediv__(self, other):
        value = other / self.value
        return Dual(value, -value * self.slope / self.value)

    def __neg__(self):
        return Dual(-self.value, -self.slope)

    def __pow__(self, exponent):
        if isinstance(exponent, Dual):
            raise TypeError("a power with a varying exponent is not defined")
        exponent = float(exponent)
        return Dual(self.value**exponent, exponent * self.value ** (exponent - 1.0) * self.slope)


class DualMath:
    """``log1p`` and ``expm1`` of pairs (and of plain arrays) with the array module ``xp``."""

    def __init__(self, xp) -> None:
        self.xp = xp

    def log1p(self, argument):
        xp = self.xp
        if not isinstance(argument, Dual):
            return xp.log1p(argument)
        return Dual(xp.log1p(argument.value), argument.slope / (1.0 + argument.value))

    def expm1(self, argument):
        xp = self.xp
        if not isinstance(argument, Dual):
            return xp.expm1(argument)
        return Dual(xp.expm1(argument.value), xp.exp(argument.value) * argument.slope)


def partials(energy, arguments, xp):
    """``energy(*arguments)`` and its partial derivatives in each argument.

    ``energy(*values, math=...)`` is evaluated once on the values and once per
    argument on pairs whose slope is one in that argument (zero in the others).
    """

    math = DualMath(xp)
    value = energy(*arguments, math=math)
    found = [value]
    for index in range(len(arguments)):
        seeded = [
            Dual(argument, xp.ones_like(argument) if position == index else xp.zeros_like(argument))
            for position, argument in enumerate(arguments)
        ]
        found.append(energy(*seeded, math=math).slope)
    return tuple(found)


__all__ = ["Dual", "DualMath", "partials"]
