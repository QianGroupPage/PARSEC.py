"""Launch arguments of a raw CUDA kernel, taken from its declaration.

A raw kernel launch is not checked against the kernel's parameter list.  An
argument that is missing, or of another width, shifts every later one, and
the kernel then reads whatever follows.  A launch that names its values and
has them placed by the declaration cannot fall behind a kernel whose
parameter list has moved: it raises instead.

Nothing here needs CuPy; device and host arrays are told apart from scalars
by the declaration alone.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np


_SCALAR_TYPES = {
    "int": np.int32,
    "long long": np.int64,
    "float": np.float32,
    "double": np.float64,
}
_POINTER_TYPES = {
    "int *": np.dtype(np.int32),
    "unsigned char *": np.dtype(np.uint8),
    "float *": np.dtype(np.float32),
    "double *": np.dtype(np.float64),
}


def kernel_parameters(source: str, name: str) -> tuple[tuple[str, str], ...]:
    """Return ``((C type, parameter name), ...)`` as ``source`` declares ``name``."""

    declaration = re.search(
        r"\bvoid\s+" + re.escape(name) + r"\s*\(([^)]*)\)", source
    )
    if declaration is None:
        raise ValueError(f"kernel {name} is not declared in its source")
    parameters = []
    for item in declaration.group(1).split(","):
        words = [
            word
            for word in item.replace("*", " * ").split()
            if word not in ("const", "__restrict__")
        ]
        kind = " ".join(words[:-1])
        if kind not in _SCALAR_TYPES and kind not in _POINTER_TYPES:
            raise ValueError(
                f"kernel {name} declares {words[-1]} as unsupported {kind!r}"
            )
        parameters.append((kind, words[-1]))
    return tuple(parameters)


def launch_arguments(
    parameters: tuple[tuple[str, str], ...],
    values: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Order ``values``, given by parameter name, as the kernel declares them.

    A scalar is converted to its declared C type.  An array must already hold
    the declared element type: converting it here would copy it in every
    launch.
    """

    names = [name for _, name in parameters]
    if len(values) != len(names) or any(name not in values for name in names):
        raise TypeError(
            "the launch does not fit the kernel declaration: "
            f"{sorted(set(names) ^ set(values))}"
        )
    arguments = []
    for kind, name in parameters:
        value = values[name]
        if kind in _POINTER_TYPES:
            if value.dtype != _POINTER_TYPES[kind]:
                raise TypeError(
                    f"{name} is declared {kind}, the array holds {value.dtype}"
                )
            arguments.append(value)
        else:
            arguments.append(_SCALAR_TYPES[kind](value))
    return tuple(arguments)


__all__ = ["kernel_parameters", "launch_arguments"]
