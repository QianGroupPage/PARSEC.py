"""Fortran edit descriptors used by the PARSEC molecular dynamics files.

PARSEC writes ``md_nrg.dat``, ``md_mech.dat``, ``atom.cor`` and ``movie.xyz``
with ``Fw.d``, ``Ew.d`` and ``Iw`` descriptors and a few list-directed
records, and it reads ``mdinit.dat`` back with ``Ew.d``.  The functions here
reproduce what the Fortran run-time library does with those descriptors, so
that the Python files agree with the Fortran ones character for character:

* a value is rounded from its exact binary expansion, as C ``printf`` does
  (this is what gfortran does for the default rounding mode);
* a field that is too narrow is filled with asterisks;
* a negative number that rounds to zero keeps its sign (``-0.000``);
* a three-digit exponent drops the letter ``E`` (``0.10000000+101``);
* on input blanks are ignored, an all-blank field is zero, and a field
  without a decimal point has its last ``d`` digits taken as the fraction.

List-directed output is not fixed by the Fortran standard.  The layout
reproduced here is that of gfortran: a record starts with one blank, a
character value is copied as it is, and a default integer takes twelve
columns.  All of this is checked against gfortran 13 in the tests.
"""

from __future__ import annotations

import math
import re


def _nonfinite(value: float, width: int) -> str:
    """NaN and infinity of gfortran's ``F`` and ``E`` output."""
    if math.isnan(value):
        text = "NaN"
    else:
        # 'Infinity' needs a field of more than eight characters.
        text = "Infinity" if width > 8 else "Inf"
        if value < 0.0:
            text = "-" + text
    if len(text) > width:
        return "*" * width
    return text.rjust(width)


def _fit(text: str, width: int) -> str:
    """Right-justify, dropping the optional leading zero before overflowing."""
    if len(text) > width:
        if text.startswith("0."):
            text = text[1:]
        elif text.startswith("-0."):
            text = "-" + text[2:]
    if len(text) > width:
        return "*" * width
    return text.rjust(width)


def fortran_f(value: float, width: int, decimals: int) -> str:
    """Format ``value`` with the output edit descriptor ``Fw.d``."""
    value = float(value)
    if not math.isfinite(value):
        return _nonfinite(value, width)
    return _fit("%.*f" % (decimals, value), width)


def fortran_e(value: float, width: int, decimals: int) -> str:
    """Format ``value`` with the output edit descriptor ``Ew.d``.

    The significand is ``0.d1d2...`` and the exponent has two digits; an
    exponent of three digits is written without the letter ``E``.
    """
    value = float(value)
    if not math.isfinite(value):
        return _nonfinite(value, width)
    sign = "-" if math.copysign(1.0, value) < 0.0 else ""
    if value == 0.0:
        digits = "0" * decimals
        exponent = 0
    else:
        mantissa, power = ("%.*e" % (decimals - 1, abs(value))).split("e")
        digits = mantissa.replace(".", "")
        exponent = int(power) + 1
    if abs(exponent) <= 99:
        tail = "E%+03d" % exponent
    else:
        tail = "%+04d" % exponent
    return _fit(sign + "0." + digits + tail, width)


def fortran_i(value: int, width: int) -> str:
    """Format ``value`` with the output edit descriptor ``Iw``."""
    text = "%d" % int(value)
    if len(text) > width:
        return "*" * width
    return text.rjust(width)


def list_directed_integer(value: int) -> str:
    """A default integer in a list-directed record (gfortran: twelve columns)."""
    return "%12d" % int(value)


_REAL_FIELD = re.compile(
    r"([+-]?)(\d*)(?:(\.)(\d*))?(?:[eEdDqQ]([+-]?\d+)|([+-]\d+))?"
)


def read_fortran_real(field: str, decimals: int) -> float:
    """Read one field with the input edit descriptor ``Ew.d`` (or ``Fw.d``).

    ``field`` is the ``w`` characters of the record.  A record shorter than
    the format is padded with blanks by Fortran, so an empty field is zero.
    """
    text = field.replace(" ", "").replace("\t", "")
    if not text:
        return 0.0
    lowered = text.lower().lstrip("+-")
    if lowered in ("nan", "inf", "infinity"):
        number = float(lowered)
        return -number if text.startswith("-") else number
    match = _REAL_FIELD.fullmatch(text)
    if match is None or not (match.group(2) or match.group(4)):
        raise ValueError(f"bad real number in Fortran field {field!r}")
    sign, integer, point, fraction, exponent_letter, exponent_bare = match.groups()
    exponent = int(exponent_letter or exponent_bare or 0)
    if point is None:
        # No decimal point: the rightmost ``d`` digits are the fraction.
        significand = integer
        exponent -= decimals
    else:
        fraction = fraction or ""
        significand = integer + fraction
        exponent -= len(fraction)
    return float(f"{sign}{significand or '0'}e{exponent}")


def read_fortran_integer(field: str) -> int:
    """Read one field with the input edit descriptor ``Iw``."""
    text = field.replace(" ", "").replace("\t", "")
    if not text:
        return 0
    if re.fullmatch(r"[+-]?\d+", text) is None:
        raise ValueError(f"bad integer in Fortran field {field!r}")
    return int(text)
